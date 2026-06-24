//! Deterministic dedup pre-stage (regex-normalize + run-length collapse).
//!
//! This is the highest-leverage deterministic compression lever for logs/traces:
//! production telemetry is dominated by the same line repeated thousands of times
//! (retry storms, heartbeats, per-request access logs). We collapse *consecutive*
//! lines that share a normalized template into "first + last + count", caching the
//! elided middle verbatim so the operation is fully reversible.
//!
//! Design decisions (from the 2026-06-04 eng review, outside-voice corrections):
//! - **regex-normalize, not Drain.** A fixed, ordered set of regexes masks the
//!   variable tokens (timestamps, UUIDs, IPs, hex, numbers) to form a normalized
//!   key; consecutive lines with the same key form a run. This is O(N),
//!   parameter-free, and deterministic by construction. A Drain fixed-depth tree
//!   would add a similarity threshold and parse-tree state we do not want, and a
//!   stateless-per-payload Drain throws away the only thing Drain is good at.
//! - **collapse-with-count, even for "important" lines.** We do NOT exempt
//!   error/5xx lines from collapsing. A 10k-line retry storm of the same 500 is
//!   exactly the redundancy we must compress. Keeping first + last + count keeps
//!   the error visible and the count informative; the cache keeps it lossless.
//!   (Relevance/REVIEW classification of the *kept* representative is a separate,
//!   later concern.)
//! - **stateless per payload.** No state persists across calls, so the same input
//!   always produces the same output. This preserves Polymorph's determinism
//!   guarantee.
//!
//! Composition with the token pipeline: the reduced text produced here becomes the
//! new canonical text fed to `lock_payload`. There are two reversibility layers
//! (this dedup cache + CCR array compression); both stash verbatim originals in
//! `ccr_cache` and are recoverable via `ccr::retrieve`.
//!
//! ```text
//! raw payload
//!   │ split into lines
//!   ▼
//! normalize each line ──► key (timestamps/UUIDs/IPs/hex/numbers masked)
//!   │ group CONSECUTIVE equal keys into runs
//!   ▼
//! run length >= min_run ? ──no──► emit lines verbatim
//!   │ yes
//!   ▼
//! keep head + tail verbatim; elide the middle ──► CollapsedGroup (cached)
//!   │
//!   ▼
//! reduced text (verbatim lines + "N lines elided" summaries)
//! ```

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use rusqlite::params;

use crate::db::DbHandle;

/// Ordered normalization patterns. Order matters and is fixed for determinism.
/// Each masks a class of variable token so that two lines differing only in their
/// variable parts collapse to the same key.
static PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    vec![
        // ISO-8601 / RFC3339 timestamps
        (
            Regex::new(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
                .unwrap(),
            "<TS>",
        ),
        // Apache/CLF timestamps: 27/Dec/2037:12:00:00 +0530
        (
            Regex::new(r"\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s*[+-]\d{4}").unwrap(),
            "<TS>",
        ),
        // UUIDs
        (
            Regex::new(
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            )
            .unwrap(),
            "<UUID>",
        ),
        // IPv4
        (Regex::new(r"\b\d{1,3}(?:\.\d{1,3}){3}\b").unwrap(), "<IP>"),
        // 0x-prefixed hex
        (Regex::new(r"\b0[xX][0-9a-fA-F]+\b").unwrap(), "<HEX>"),
        // long bare hex (>=16 chars: object ids, hashes, span ids)
        (Regex::new(r"\b[0-9a-fA-F]{16,}\b").unwrap(), "<HEX>"),
        // bare numbers (ints/floats) — last so it doesn't chew the classes above
        (Regex::new(r"\b\d+(?:\.\d+)?\b").unwrap(), "<NUM>"),
    ]
});

#[derive(Debug, Clone, Copy)]
pub struct DedupOpts {
    /// Lines to keep verbatim at the head of a collapsed run.
    pub head_keep: usize,
    /// Lines to keep verbatim at the tail of a collapsed run.
    pub tail_keep: usize,
    /// Minimum run length before we collapse. Must leave a non-empty middle, i.e.
    /// `min_run >= head_keep + tail_keep + 1`.
    pub min_run: usize,
}

impl Default for DedupOpts {
    fn default() -> Self {
        Self {
            head_keep: 1,
            tail_keep: 1,
            min_run: 3,
        }
    }
}

/// A collapsed run: the verbatim middle lines that were elided, plus bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollapsedGroup {
    /// Stable index of this group within the payload (placeholder reference).
    pub group_idx: usize,
    /// The normalized template key shared by every line in the run.
    pub key: String,
    /// The elided (middle) lines, verbatim and in order. Reversible source of truth.
    pub elided: Vec<String>,
}

/// One unit of the reduced output: either a verbatim line or a collapsed run.
/// This is the STRUCTURAL source of truth for reconstruction — `reconstruct`
/// walks these units rather than re-parsing the rendered `reduced` string, so a
/// round-trip is exact even when an original line happens to equal the cosmetic
/// summary sentinel (which would otherwise be silent data corruption).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Unit {
    Verbatim(String),
    Collapsed(usize), // index into DedupPlan.groups
}

/// Result of the pure dedup pass.
#[derive(Debug, Clone)]
pub struct DedupPlan {
    /// The reduced payload: verbatim kept lines + deterministic "elided" summaries.
    pub reduced: String,
    /// Elided middles, one per collapsed run. Index == `group_idx`.
    pub groups: Vec<CollapsedGroup>,
    /// Whether the original payload ended in a trailing newline (for exact rebuild).
    pub trailing_newline: bool,
    /// Structural unit list; reconstruction reads this, not the rendered string.
    units: Vec<Unit>,
}

impl DedupPlan {
    pub fn elided_line_count(&self) -> usize {
        self.groups.iter().map(|g| g.elided.len()).sum()
    }
}

/// Normalize a single line into its template key by masking variable tokens.
/// Deterministic: fixed pattern set applied in fixed order.
pub fn normalize_line(line: &str) -> String {
    let mut out = line.to_string();
    for (re, repl) in PATTERNS.iter() {
        out = re.replace_all(&out, *repl).into_owned();
    }
    out
}

/// Deterministic summary line injected in place of an elided run middle.
/// Carries the group index, count, and key so the output is self-describing and
/// the persistence layer can attach a `cache_id` for retrieval.
fn summary_line(group_idx: usize, elided_count: usize, key: &str) -> String {
    format!("\u{27EA}polymorph:elided idx={group_idx} lines={elided_count} key={key:?}\u{27EB}")
}

/// Pure, deterministic dedup pass. No DB, no RNG, no clock.
///
/// Splits `text` on `\n`, groups consecutive lines sharing a normalized key, and
/// collapses runs `>= opts.min_run` into head + summary + tail. Returns the reduced
/// text and the elided middles (for caching / round-trip).
pub fn dedup_plan(text: &str, opts: DedupOpts) -> DedupPlan {
    let trailing_newline = text.ends_with('\n');
    // `lines()` drops the trailing empty segment; we restore the trailing newline
    // on render. An empty input yields no lines.
    let lines: Vec<&str> = if text.is_empty() {
        Vec::new()
    } else {
        text.split('\n').collect::<Vec<_>>()
    };
    // split('\n') on "a\n" yields ["a", ""]; drop that synthetic trailing empty.
    let lines: Vec<&str> = if trailing_newline {
        let mut l = lines;
        l.pop();
        l
    } else {
        lines
    };

    let keys: Vec<String> = lines.iter().map(|l| normalize_line(l)).collect();

    let mut units: Vec<Unit> = Vec::new();
    let mut groups: Vec<CollapsedGroup> = Vec::new();

    let head = opts.head_keep.max(1);
    let tail = opts.tail_keep;
    let min_run = opts.min_run.max(head + tail + 1);

    let mut i = 0;
    while i < lines.len() {
        // Extend a run of identical keys.
        let mut j = i + 1;
        while j < lines.len() && keys[j] == keys[i] {
            j += 1;
        }
        let run_len = j - i;

        if run_len >= min_run {
            // head verbatim
            for line in lines.iter().skip(i).take(head) {
                units.push(Unit::Verbatim((*line).to_string()));
            }
            // middle elided
            let mid_start = i + head;
            let mid_end = j - tail;
            let elided: Vec<String> = (mid_start..mid_end).map(|k| lines[k].to_string()).collect();
            let group_idx = groups.len();
            groups.push(CollapsedGroup {
                group_idx,
                key: keys[i].clone(),
                elided,
            });
            units.push(Unit::Collapsed(group_idx));
            // tail verbatim
            for line in lines.iter().take(j).skip(j - tail) {
                units.push(Unit::Verbatim((*line).to_string()));
            }
        } else {
            for line in lines.iter().take(j).skip(i) {
                units.push(Unit::Verbatim((*line).to_string()));
            }
        }
        i = j;
    }

    let reduced = render_reduced(&units, &groups, trailing_newline);

    DedupPlan {
        reduced,
        groups,
        trailing_newline,
        units,
    }
}

/// Render the human/LLM-facing reduced text from the structural units. The
/// summary line is cosmetic only; reconstruction never parses it back.
fn render_reduced(units: &[Unit], groups: &[CollapsedGroup], trailing_newline: bool) -> String {
    let mut out: Vec<String> = Vec::with_capacity(units.len());
    for u in units {
        match u {
            Unit::Verbatim(s) => out.push(s.clone()),
            Unit::Collapsed(idx) => {
                let g = &groups[*idx];
                out.push(summary_line(*idx, g.elided.len(), &g.key));
            }
        }
    }
    let mut s = out.join("\n");
    if trailing_newline {
        s.push('\n');
    }
    s
}

/// Reconstruct the original payload exactly. Walks the structural unit list
/// (NOT the rendered string), so the round-trip is byte-for-byte even when an
/// original line equals the cosmetic summary sentinel. Verbatim units emit their
/// stored line; collapsed units emit their group's elided middle in order.
pub fn reconstruct(plan: &DedupPlan) -> String {
    let mut out: Vec<String> = Vec::new();
    for u in &plan.units {
        match u {
            Unit::Verbatim(s) => out.push(s.clone()),
            Unit::Collapsed(idx) => out.extend(plan.groups[*idx].elided.iter().cloned()),
        }
    }
    let mut s = out.join("\n");
    if plan.trailing_newline {
        s.push('\n');
    }
    s
}

/// Persist each collapsed group's elided lines into `ccr_cache` and return a
/// manifest mapping `group_idx -> (cache_id, elided_count)`. Writes are done in a
/// SINGLE transaction through the DB actor to avoid write amplification / holding
/// the actor open for thousands of separate calls.
pub fn persist_groups(
    plan: &DedupPlan,
    db: &DbHandle,
    created_at: i64,
) -> Result<Vec<(usize, String, usize)>> {
    if plan.groups.is_empty() {
        return Ok(Vec::new());
    }
    // Pre-serialize outside the actor closure (keep the worker thread cheap).
    // Propagate serialization errors: never store a payload we cannot read back
    // (an empty blob would be a silent, unrecoverable data loss on retrieval).
    let rows: Vec<(usize, String, Vec<u8>, usize)> = plan
        .groups
        .iter()
        .map(|g| {
            let id = uuid::Uuid::new_v4().to_string();
            let payload = serde_json::to_vec(&g.elided)?;
            Ok((g.group_idx, id, payload, g.elided.len()))
        })
        .collect::<Result<Vec<_>>>()?;

    let rows_for_db = rows.clone();
    db.call(move |conn| {
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO ccr_cache (id, payload, omitted_count, created_at) VALUES (?1, ?2, ?3, ?4)",
            )?;
            for (_idx, id, payload, count) in &rows_for_db {
                stmt.execute(params![id, payload, *count as i64, created_at])?;
            }
        }
        tx.commit()?;
        Ok(())
    })?;

    Ok(rows
        .into_iter()
        .map(|(idx, id, _payload, count)| (idx, id, count))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    #[test]
    fn normalize_masks_variable_tokens() {
        let a = normalize_line("233.223.117.90 - - [27/Dec/2037:12:00:00 +0530] \"GET /x\" 200 42");
        let b = normalize_line("162.253.4.179 - - [27/Dec/2037:13:00:00 +0530] \"GET /x\" 200 99");
        assert_eq!(
            a, b,
            "lines differing only in IP/timestamp/number must share a key"
        );
        assert!(a.contains("<IP>") && a.contains("<TS>") && a.contains("<NUM>"));
    }

    #[test]
    fn collapses_a_run_and_round_trips() {
        let mut lines = Vec::new();
        for i in 0..50 {
            lines.push(format!("10.0.0.{} - heartbeat ok {}", i % 5, i));
        }
        let text = lines.join("\n");
        let plan = dedup_plan(&text, DedupOpts::default());
        // All 50 share the same normalized key -> one collapsed group.
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].elided.len(), 50 - 2); // head=1 + tail=1 kept
                                                         // Reduced is much shorter.
        assert!(plan.reduced.lines().count() < 5);
        // Round-trip is exact.
        assert_eq!(reconstruct(&plan), text);
    }

    #[test]
    fn distinct_lines_are_not_collapsed() {
        let text = "alpha 1\nbeta 2\ngamma 3\ndelta 4";
        let plan = dedup_plan(text, DedupOpts::default());
        assert!(plan.groups.is_empty());
        assert_eq!(plan.reduced, text);
        assert_eq!(reconstruct(&plan), text);
    }

    #[test]
    fn deterministic_across_calls_and_independent_of_history() {
        let text = (0..30)
            .map(|i| format!("req id=abc-{} status=200 took={}ms", i, i))
            .collect::<Vec<_>>()
            .join("\n");
        // Run twice; prime with unrelated input in between to prove statelessness.
        let p1 = dedup_plan(&text, DedupOpts::default());
        let _ = dedup_plan("unrelated\nnoise\nhere", DedupOpts::default());
        let p2 = dedup_plan(&text, DedupOpts::default());
        assert_eq!(p1.reduced, p2.reduced);
        assert_eq!(p1.groups, p2.groups);
    }

    #[test]
    fn repeated_error_line_collapses_but_first_and_last_survive() {
        // A retry storm: the same 5xx line 1000x. Must collapse, but the error
        // must remain visible (first + last kept).
        let err = "ERROR 503 upstream payment timeout";
        let text = std::iter::repeat_n(err, 1000)
            .collect::<Vec<_>>()
            .join("\n");
        let plan = dedup_plan(&text, DedupOpts::default());
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].elided.len(), 998);
        assert!(plan.reduced.contains(err), "error line must stay visible");
        assert!(plan.reduced.contains("lines=998"), "count must be reported");
        assert_eq!(reconstruct(&plan), text);
    }

    #[test]
    fn trailing_newline_preserved() {
        let text = "x x x\nx x x\nx x x\nx x x\n";
        let plan = dedup_plan(text, DedupOpts::default());
        assert!(plan.reduced.ends_with('\n'));
        assert_eq!(reconstruct(&plan), text);
    }

    #[test]
    fn empty_and_single_line() {
        let p0 = dedup_plan("", DedupOpts::default());
        assert_eq!(p0.reduced, "");
        assert!(p0.groups.is_empty());
        assert_eq!(reconstruct(&p0), "");

        let p1 = dedup_plan("only one line", DedupOpts::default());
        assert_eq!(p1.reduced, "only one line");
        assert!(p1.groups.is_empty());
    }

    #[test]
    fn interleaved_runs_only_collapse_consecutive() {
        // A B A B ... never collapses (no consecutive run); A A A ... B B B does.
        let text = "A x\nB x\nA x\nB x\nA x\nB x";
        let plan = dedup_plan(text, DedupOpts::default());
        assert!(plan.groups.is_empty(), "interleaved keys form no run");
        assert_eq!(reconstruct(&plan), text);
    }

    #[test]
    fn persist_groups_round_trips_through_cache() {
        let pool = db::test_pool().unwrap();
        let text = std::iter::repeat_n("heartbeat ok 1", 40)
            .collect::<Vec<_>>()
            .join("\n");
        let plan = dedup_plan(&text, DedupOpts::default());
        let manifest = persist_groups(&plan, &pool, 0).unwrap();
        assert_eq!(manifest.len(), 1);
        let (_idx, cache_id, count) = &manifest[0];
        assert_eq!(*count, 38);
        // Recover the elided middle verbatim via the existing CCR retrieve path.
        let recovered = crate::ccr::retrieve(cache_id, &pool).unwrap();
        let recovered: Vec<String> = serde_json::from_value(recovered).unwrap();
        assert_eq!(recovered, plan.groups[0].elided);
    }

    #[test]
    fn reconstruct_survives_sentinel_collision() {
        // Regression: an original line byte-identical to the cosmetic summary
        // sentinel must NOT be reinterpreted as a collapse marker on rebuild.
        // (Adversarial finding, 2026-06-05 /review.)
        let input = "\u{27EA}polymorph:elided idx=0 lines=2 key=\"x\"\u{27EB}\n\
                     UNIQUE-MARKER-A\n\
                     rep\nrep\nrep\nrep\nrep";
        let plan = dedup_plan(input, DedupOpts::default());
        assert_eq!(
            reconstruct(&plan),
            input,
            "round-trip must be exact despite sentinel collision"
        );
    }

    #[test]
    fn reconstruct_when_input_is_only_a_fake_summary_line() {
        let input = "\u{27EA}polymorph:elided idx=99 lines=5 key=\"boom\"\u{27EB}";
        let plan = dedup_plan(input, DedupOpts::default());
        assert!(plan.groups.is_empty());
        assert_eq!(reconstruct(&plan), input);
    }

    #[test]
    fn crlf_line_endings_round_trip() {
        let input = "a x\r\na x\r\na x\r\na x\r\n";
        let plan = dedup_plan(input, DedupOpts::default());
        assert_eq!(reconstruct(&plan), input);
    }

    #[test]
    fn multibyte_lines_round_trip() {
        let input = "héllo 🌍 1\nhéllo 🌍 2\nhéllo 🌍 3\nhéllo 🌍 4\nhéllo 🌍 5";
        let plan = dedup_plan(input, DedupOpts::default());
        assert_eq!(reconstruct(&plan), input);
    }

    #[test]
    fn persist_empty_plan_is_noop() {
        let pool = db::test_pool().unwrap();
        let plan = dedup_plan("a\nb\nc", DedupOpts::default());
        let manifest = persist_groups(&plan, &pool, 0).unwrap();
        assert!(manifest.is_empty());
    }
}

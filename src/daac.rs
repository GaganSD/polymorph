use anyhow::Result;
use daachorse::DoubleArrayAhoCorasick;

use crate::tokenizer;

/// DAAC scanner over a BPE token-id stream.
///
/// Keywords are tokenized once at build time. At scan time the document token
/// stream is serialized to little-endian u32 bytes and traversed in a single
/// forward pass with no backtracking.
pub struct DaacScanner {
    automaton: Option<DoubleArrayAhoCorasick<u32>>,
    /// Reused haystack buffer — cleared and refilled each scan; no per-scan Vec alloc.
    haystack: Vec<u8>,
}

impl DaacScanner {
    pub fn build(keywords: &[String]) -> Result<Self> {
        if keywords.is_empty() {
            return Ok(Self {
                automaton: None,
                haystack: Vec::new(),
            });
        }

        let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(keywords.len());
        for kw in keywords {
            let (ids, _) = tokenizer::token_spans(kw)?;
            if ids.is_empty() {
                continue;
            }
            let mut bytes = Vec::with_capacity(ids.len() * 4);
            for id in &ids {
                bytes.extend_from_slice(&id.to_le_bytes());
            }
            patterns.push(bytes);
        }

        if patterns.is_empty() {
            return Ok(Self {
                automaton: None,
                haystack: Vec::new(),
            });
        }

        let automaton = DoubleArrayAhoCorasick::<u32>::new(patterns)
            .map_err(|e| anyhow::anyhow!("daachorse build failed: {e}"))?;
        Ok(Self {
            automaton: Some(automaton),
            haystack: Vec::new(),
        })
    }

    /// Single-pass scan; returns merged token-index intervals `[start, end)`.
    pub fn scan(&mut self, token_ids: &[u32]) -> Vec<(usize, usize)> {
        let Some(aut) = &self.automaton else {
            return Vec::new();
        };

        self.haystack.clear();
        self.haystack.reserve(token_ids.len() * 4);
        for id in token_ids {
            self.haystack.extend_from_slice(&id.to_le_bytes());
        }

        let mut raw: Vec<(usize, usize)> = Vec::new();
        for m in aut.find_overlapping_iter(&self.haystack) {
            debug_assert_eq!(m.start() % 4, 0);
            debug_assert_eq!(m.end() % 4, 0);
            raw.push((m.start() / 4, m.end() / 4));
        }

        crate::locking::sort_and_merge(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::token_spans;

    #[test]
    fn finds_keyword_in_token_stream() {
        let text = r#"{"api_key":"sk-prod-123","note":"hello"}"#;
        let (ids, _) = token_spans(text).unwrap();
        let mut scanner = DaacScanner::build(&["sk-prod-123".to_string()]).unwrap();
        let hits = scanner.scan(&ids);
        assert!(!hits.is_empty(), "should find keyword");
        for (s, e) in &hits {
            assert!(s < e);
            assert!(*e <= ids.len());
        }
    }

    #[test]
    fn empty_keywords_returns_no_matches() {
        let (ids, _) = token_spans("hello").unwrap();
        let mut scanner = DaacScanner::build(&[]).unwrap();
        assert!(scanner.scan(&ids).is_empty());
    }

    #[test]
    fn no_match_when_absent() {
        let (ids, _) = token_spans("nothing to see here").unwrap();
        let mut scanner = DaacScanner::build(&["sk-prod-XXX".to_string()]).unwrap();
        assert!(scanner.scan(&ids).is_empty());
    }

    #[test]
    fn single_pass_no_backtracking() {
        let (ids, _) = token_spans("abc def abc").unwrap();
        let mut scanner = DaacScanner::build(&["abc".to_string()]).unwrap();
        let hits = scanner.scan(&ids);
        assert!(!hits.is_empty());
    }
}

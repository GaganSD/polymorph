use anyhow::Result;
use daachorse::DoubleArrayAhoCorasick;

use crate::tokens;

/// DAAC scanner that searches for keyword patterns over an integer token stream.
///
/// We tokenize each keyword with the same BPE used for the input document, then
/// build a Double-Array Aho-Corasick automaton over those `Vec<u32>` patterns
/// serialized as little-endian 4-byte chunks. At scan time, we serialize the
/// document's token stream the same way and run the automaton over those bytes.
/// Byte offsets `b/4` correspond exactly to token indices because every token
/// id occupies exactly 4 bytes — there's no false alignment in the middle of an
/// id because every match start/end is a multiple of 4 by construction (each
/// pattern is also 4-byte aligned).
pub struct DaacScanner {
    automaton: Option<DoubleArrayAhoCorasick<u32>>,
}

impl DaacScanner {
    pub fn build(keywords: &[String]) -> Result<Self> {
        if keywords.is_empty() {
            return Ok(Self { automaton: None });
        }

        let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(keywords.len());
        for kw in keywords {
            let (ids, _) = tokens::token_spans(kw)?;
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
            return Ok(Self { automaton: None });
        }

        let automaton = DoubleArrayAhoCorasick::<u32>::new(patterns)
            .map_err(|e| anyhow::anyhow!("daachorse build failed: {e}"))?;
        Ok(Self {
            automaton: Some(automaton),
        })
    }

    /// Returns matched token-index intervals `[start, end)`.
    pub fn scan(&self, token_ids: &[u32]) -> Vec<(usize, usize)> {
        let Some(aut) = &self.automaton else {
            return Vec::new();
        };

        let mut haystack: Vec<u8> = Vec::with_capacity(token_ids.len() * 4);
        for id in token_ids {
            haystack.extend_from_slice(&id.to_le_bytes());
        }

        let mut out: Vec<(usize, usize)> = Vec::new();
        for m in aut.find_overlapping_iter(&haystack) {
            // Both endpoints are guaranteed to be multiples of 4: patterns are
            // 4-byte-aligned token-id sequences, and the haystack is too.
            debug_assert_eq!(m.start() % 4, 0);
            debug_assert_eq!(m.end() % 4, 0);
            out.push((m.start() / 4, m.end() / 4));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::token_spans;

    #[test]
    fn finds_keyword_in_token_stream() {
        let text = r#"{"api_key":"sk-prod-123","note":"hello"}"#;
        let (ids, _) = token_spans(text).unwrap();
        let scanner = DaacScanner::build(&["sk-prod-123".to_string()]).unwrap();
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
        let scanner = DaacScanner::build(&[]).unwrap();
        assert!(scanner.scan(&ids).is_empty());
    }

    #[test]
    fn no_match_when_absent() {
        let (ids, _) = token_spans("nothing to see here").unwrap();
        let scanner = DaacScanner::build(&["sk-prod-XXX".to_string()]).unwrap();
        assert!(scanner.scan(&ids).is_empty());
    }
}

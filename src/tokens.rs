use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use tiktoken_rs::CoreBPE;

static BPE: OnceCell<CoreBPE> = OnceCell::new();

fn bpe() -> Result<&'static CoreBPE> {
    BPE.get_or_try_init(tiktoken_rs::cl100k_base)
        .map_err(|e| anyhow!("failed to load cl100k_base: {e}"))
}

/// Tokenize `text` and return both the token ID stream and a parallel array of
/// (start_byte, end_byte) spans into the original input.
///
/// cl100k uses byte-level BPE, so concatenating `decode_single_token_bytes` for
/// each id reproduces the input byte-for-byte — that's what lets us assign
/// each token a precise byte range without relying on a tokenizer API for
/// offsets.
pub fn token_spans(text: &str) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
    let bpe = bpe()?;
    let ids: Vec<u32> = bpe.encode_ordinary(text);
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(ids.len());

    let mut cursor: usize = 0;
    for bytes in bpe._decode_native_and_split(ids.clone()) {
        let start = cursor;
        cursor += bytes.len();
        spans.push((start, cursor));
    }

    if cursor != text.len() {
        return Err(anyhow!(
            "byte-span reconstruction mismatch: cursor={cursor} text.len()={}",
            text.len()
        ));
    }

    Ok((ids, spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_spans_round_trip_simple() {
        let s = "hello world";
        let (ids, spans) = token_spans(s).unwrap();
        assert_eq!(ids.len(), spans.len());
        assert!(!ids.is_empty());
        assert_eq!(spans.first().unwrap().0, 0);
        assert_eq!(spans.last().unwrap().1, s.len());
        for w in spans.windows(2) {
            assert_eq!(w[0].1, w[1].0, "spans must be contiguous");
        }
    }

    #[test]
    fn token_spans_unicode() {
        let s = "héllo🌍world";
        let (ids, spans) = token_spans(s).unwrap();
        assert_eq!(ids.len(), spans.len());
        assert_eq!(spans.last().unwrap().1, s.len());
    }

    #[test]
    fn token_spans_json_like() {
        let s = r#"{"k":"v"}"#;
        let (ids, spans) = token_spans(s).unwrap();
        assert_eq!(spans.last().unwrap().1, s.len());
        assert_eq!(ids.len(), spans.len());
    }
}

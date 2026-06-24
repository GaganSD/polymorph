//! Bounded async stdin wrapper.
//!
//! rmcp's stdio transport reads newline-delimited JSON-RPC messages directly
//! from `tokio::io::stdin()`. The SDK provides no per-message size cap, so we
//! interpose a `BoundedAsyncRead` that counts bytes since the last newline and
//! returns an `io::Error` if a single line exceeds `MAX_PAYLOAD_BYTES`. This
//! defeats zip-bomb-style egress without allocating beyond the bound.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, ReadBuf};

use crate::io_guard::MAX_PAYLOAD_BYTES;

pub struct BoundedAsyncRead<R> {
    inner: R,
    /// Bytes consumed on the current (in-progress) line, not counting the trailing '\n'.
    line_bytes: u64,
}

impl<R> BoundedAsyncRead<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            line_bytes: 0,
        }
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for BoundedAsyncRead<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let before = buf.filled().len();
        let res = Pin::new(&mut self.inner).poll_read(cx, buf);
        if let Poll::Ready(Ok(())) = &res {
            let new = &buf.filled()[before..];
            let mut tripped = false;
            for &b in new {
                if b == b'\n' {
                    self.line_bytes = 0;
                } else {
                    self.line_bytes = self.line_bytes.saturating_add(1);
                    if self.line_bytes > MAX_PAYLOAD_BYTES {
                        tripped = true;
                        break;
                    }
                }
            }
            if tripped {
                buf.set_filled(before);
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("incoming message exceeds {MAX_PAYLOAD_BYTES} bytes \u{2014} refusing"),
                )));
            }
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    #[tokio::test]
    async fn passes_small_messages_through() {
        let data: &[u8] = b"hello\nworld\n";
        let mut r = BoundedAsyncRead::new(data);
        let mut out = Vec::new();
        r.read_to_end(&mut out).await.unwrap();
        assert_eq!(out, data);
    }

    #[tokio::test]
    async fn resets_counter_on_newline() {
        // Two lines, each smaller than the cap; together larger than the cap.
        let small = vec![b'a'; (MAX_PAYLOAD_BYTES / 2) as usize];
        let mut data = small.clone();
        data.push(b'\n');
        data.extend_from_slice(&small);
        data.push(b'\n');
        let mut r = BoundedAsyncRead::new(&data[..]);
        let mut out = Vec::new();
        r.read_to_end(&mut out).await.unwrap();
        assert_eq!(out.len(), data.len());
    }

    #[tokio::test]
    async fn rejects_oversize_single_line() {
        let huge = vec![b'a'; (MAX_PAYLOAD_BYTES as usize) + 100];
        let mut r = BoundedAsyncRead::new(&huge[..]);
        let mut out = Vec::new();
        let err = r.read_to_end(&mut out).await.expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(format!("{err}").contains("exceeds"));
    }

    #[tokio::test]
    async fn rejects_chunked_oversize_line() {
        use tokio::io::AsyncWriteExt;
        let (mut w, r) = tokio::io::duplex(8 * 1024);
        let handle = tokio::spawn(async move {
            // Write the oversize line in many small chunks; no newline ever.
            let chunk = vec![b'b'; 64 * 1024];
            let mut written: u64 = 0;
            while written <= MAX_PAYLOAD_BYTES + 8 * 1024 {
                if w.write_all(&chunk).await.is_err() {
                    break;
                }
                written += chunk.len() as u64;
            }
        });
        let mut r = BoundedAsyncRead::new(r);
        let mut buf = vec![0u8; 4096];
        let mut total = 0u64;
        let err = loop {
            match r.read(&mut buf).await {
                Ok(0) => panic!("EOF before cap trip; read {total}"),
                Ok(n) => total += n as u64,
                Err(e) => break e,
            }
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        handle.abort();
    }
}

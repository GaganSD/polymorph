use std::path::{Path, PathBuf};

/// Env var pointing at the exported `model.onnx`.
pub const LAMR_MODEL_ENV: &str = "POLYMORPH_LAMR_MODEL";

/// Expand a leading `~/` in user-facing config paths. Environment variables are
/// passed verbatim otherwise so absolute and relative paths keep their meaning.
pub fn expand_home_path(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

/// Install-time status for the configured LaMR model path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LamrModelStatus {
    Unset,
    Empty,
    Missing { raw: String, resolved: PathBuf },
    Found(PathBuf),
}

impl LamrModelStatus {
    pub fn from_env() -> Self {
        let raw = match std::env::var(LAMR_MODEL_ENV) {
            Ok(value) => value,
            Err(_) => return Self::Unset,
        };
        if raw.is_empty() {
            return Self::Empty;
        }
        let resolved = expand_home_path(&raw);
        if resolved.exists() {
            Self::Found(resolved)
        } else {
            Self::Missing { raw, resolved }
        }
    }

    pub fn found_path(&self) -> Option<&Path> {
        match self {
            Self::Found(path) => Some(path),
            Self::Unset | Self::Empty | Self::Missing { .. } => None,
        }
    }

    pub fn inactive_warning(&self) -> Option<String> {
        match self {
            Self::Missing { raw, resolved } => Some(format!(
                "[lamr] {LAMR_MODEL_ENV} is set to {raw:?}, resolved to {}, but the file does not exist; leaving compress_log in deterministic mode (used_model=false)",
                resolved.display()
            )),
            Self::Empty => Some(format!(
                "[lamr] {LAMR_MODEL_ENV} is set but empty; leaving compress_log in deterministic mode (used_model=false)"
            )),
            Self::Unset | Self::Found(_) => None,
        }
    }
}

pub fn lamr_model_status() -> LamrModelStatus {
    LamrModelStatus::from_env()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lamr_model_status_reports_unset() {
        let prev = std::env::var(LAMR_MODEL_ENV).ok();
        std::env::remove_var(LAMR_MODEL_ENV);
        assert_eq!(lamr_model_status(), LamrModelStatus::Unset);
        if let Some(v) = prev {
            std::env::set_var(LAMR_MODEL_ENV, v);
        }
    }

    #[test]
    fn lamr_model_status_expands_home_and_reports_missing() {
        let prev = std::env::var(LAMR_MODEL_ENV).ok();
        let file_name = format!("no-such-model-{}.onnx", uuid::Uuid::new_v4());
        let raw = format!("~/.polymorph/{file_name}");
        std::env::set_var(LAMR_MODEL_ENV, &raw);
        let home = dirs::home_dir().expect("home dir");
        assert_eq!(
            lamr_model_status(),
            LamrModelStatus::Missing {
                raw,
                resolved: home.join(".polymorph").join(file_name)
            }
        );
        match prev {
            Some(v) => std::env::set_var(LAMR_MODEL_ENV, v),
            None => std::env::remove_var(LAMR_MODEL_ENV),
        }
    }
}

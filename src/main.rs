use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let grammars_dir = resolve_grammars_dir();

    if args.iter().any(|a| a == "--selftest") {
        return polymorph::selftest::run(&grammars_dir);
    }

    polymorph::mcp::serve(grammars_dir)
}

fn resolve_grammars_dir() -> PathBuf {
    if let Ok(env) = std::env::var("POLYMORPH_GRAMMARS_DIR") {
        return PathBuf::from(env);
    }
    let exe = std::env::current_exe().ok();
    if let Some(exe) = exe {
        // dev: target/debug/polymorph-mcp -> ../../grammars
        let candidate = exe
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(|p| p.join("grammars"));
        if let Some(c) = candidate {
            if c.exists() {
                return c;
            }
        }
    }
    PathBuf::from("grammars")
}

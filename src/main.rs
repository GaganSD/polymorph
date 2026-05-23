use anyhow::Result;

use polymorph::{db, lcm, mcp};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let grammars_dir = polymorph::resolve_grammars_dir();

    if let Some(pos) = args.iter().position(|a| a == "--demo") {
        let kind = args.get(pos + 1).map(String::as_str).unwrap_or("");
        return polymorph::demo::run(kind, &grammars_dir);
    }

    if args.iter().any(|a| a == "--selftest") {
        return polymorph::selftest::run(&grammars_dir);
    }

    // Set up shared state: SQLite pool + LCM archiver worker.
    let db_path = db::default_path()?;
    let pool = db::open_pool(&db_path)?;
    let archiver = lcm::Archiver::spawn(pool.clone());

    let state = mcp::AppState {
        pool,
        archiver,
        grammars_dir,
    };
    mcp::serve(state)
}

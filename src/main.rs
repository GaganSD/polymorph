use anyhow::Result;

use polymorph::{db, mcp::PolymorphServer, transport::BoundedAsyncRead};
use rmcp::ServiceExt;

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

    if let Some(pos) = args.iter().position(|a| a == "--bench") {
        let dir = args
            .get(pos + 1)
            .map(String::as_str)
            .unwrap_or("data/raw");
        let chunk_kb = args.get(pos + 2).and_then(|s| s.parse::<usize>().ok());
        let max_mb = args.get(pos + 3).and_then(|s| s.parse::<usize>().ok());
        return polymorph::bench::run(dir, chunk_kb, max_mb);
    }

    let db_path = db::default_path()?;
    let db = db::open_pool(&db_path)?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let server = PolymorphServer::new(db, grammars_dir);
        let stdin = BoundedAsyncRead::new(tokio::io::stdin());
        let stdout = tokio::io::stdout();
        let running = server.serve((stdin, stdout)).await?;
        running.waiting().await?;
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

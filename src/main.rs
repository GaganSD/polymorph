use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let grammars_dir = polymorph::resolve_grammars_dir();

    if args.iter().any(|a| a == "--selftest") {
        return polymorph::selftest::run(&grammars_dir);
    }

    polymorph::mcp::serve(grammars_dir)
}

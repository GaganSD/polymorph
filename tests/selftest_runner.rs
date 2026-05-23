use std::path::PathBuf;

fn grammars_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars")
}

#[test]
fn selftest_passes() {
    polymorph::selftest::run(&grammars_dir()).expect("selftest should pass");
}

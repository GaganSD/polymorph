use polymorph::demo;
use std::path::PathBuf;

fn grammars_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars")
}

#[test]
fn demo_lcm_loop_runs_to_completion() {
    demo::run("lcm-loop", &grammars_dir()).expect("lcm-loop demo should succeed");
}

#[test]
fn demo_ccr_runs_to_completion() {
    demo::run("ccr", &grammars_dir()).expect("ccr demo should succeed");
}

#[test]
fn demo_unknown_kind_errors() {
    let err = demo::run("not-a-kind", &grammars_dir()).unwrap_err();
    assert!(format!("{err}").contains("unknown demo"));
}

#[test]
fn demo_empty_kind_errors() {
    let err = demo::run("", &grammars_dir()).unwrap_err();
    assert!(format!("{err}").contains("missing demo kind"));
}

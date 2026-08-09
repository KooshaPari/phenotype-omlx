use std::process::Command;

#[test]
fn smoke_build() {
    let output = Command::new("cargo")
        .arg("build")
        .output()
        .expect("Failed to execute cargo build");
    assert!(output.status.success(), "cargo build failed");
}

#[test]
fn smoke_test_compile() {
    let output = Command::new("cargo")
        .arg("test")
        .arg("--no-run")
        .output()
        .expect("Failed to execute cargo test");
    assert!(output.status.success(), "cargo test --no-run failed");
}

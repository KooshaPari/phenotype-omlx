fn main() {
    let mut build = cc::Build::new();
    build
        .file("c-src/turbo_quant.c")
        .include("c-src")
        .include("../native-abi/include")
        .flag("-O3")
        .flag("-march=native")
        .warnings(false);

    // Rust's macOS targets default to a deployment target of 11.0, while
    // clang otherwise stamps C objects with the host SDK version.  Mixing
    // those objects produces linker warnings (and can become an error when
    // the minimum target is enforced).  Honour an explicit target when the
    // caller provides one; otherwise match the Rust macOS baseline.
    if std::env::var_os("CARGO_CFG_TARGET_OS").as_deref() == Some(std::ffi::OsStr::new("macos")) {
        let target =
            std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".to_owned());
        build.flag(&format!("-mmacosx-version-min={target}"));
    }

    build.compile("turbo_quant_c");
    println!("cargo:rerun-if-changed=c-src/turbo_quant.c");
    println!("cargo:rerun-if-changed=c-src/turbo_quant.h");
    println!("cargo:rerun-if-changed=../native-abi/include/abi_v1.h");
}

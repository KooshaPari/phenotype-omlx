fn main() {
    cc::Build::new()
        .file("c-src/turbo_quant.c")
        .include("c-src")
        .include("../native-abi/include")
        .flag("-O3")
        .flag("-march=native")
        .warnings(false)
        .compile("turbo_quant_c");
    println!("cargo:rerun-if-changed=c-src/turbo_quant.c");
    println!("cargo:rerun-if-changed=c-src/turbo_quant.h");
    println!("cargo:rerun-if-changed=../native-abi/include/abi_v1.h");
}

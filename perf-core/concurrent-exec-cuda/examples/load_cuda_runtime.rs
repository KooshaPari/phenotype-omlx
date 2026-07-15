//! Example: dlopen libcudart + libphenotype_omlx_cuda and resolve a kernel.
//!
//! Run with:
//!   cargo run --example load_cuda_runtime --features cuda
//!
//! Requires the .so to exist in the working directory:
//!   cd cuda && ./build.sh   # produces libphenotype_omlx_cuda.so

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use concurrent_exec_cuda::loader::{open_kernels, open_runtime, resolve};

    let rt = open_runtime()?;
    println!("loaded libcudart OK");

    let kup = open_kernels()?;
    println!("loaded libphenotype_omlx_cuda.so OK");

    // The kernels are declared `extern "C"` in cuda/kernels/*.cu so we can
    // dlsym them by raw name. The cast back to a typed fn pointer is the
    // caller's responsibility (see cuLaunchKernel in cuda/loader.rs).
    let handle = resolve(&kup, "latentmas_fanout_kernel")?;
    println!("resolved latentmas_fanout_kernel @ {:p}", handle);

    let _ = rt; // keep rt alive; cuLaunchKernel would chain through it
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example requires: cargo run --features cuda");
    std::process::exit(2);
}
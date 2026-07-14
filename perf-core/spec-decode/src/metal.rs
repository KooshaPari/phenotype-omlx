//! Optional Metal backend for spec-decode.
//! Only compiled when `metal` feature is enabled.

#[cfg(feature = "metal")]
pub mod kernels {
    //! Wrappers around Metal compute kernels for KV-cache quantization
    //! and tree-attention score aggregation. Actual Metal kernels live
    //! in `.metallib` form and are loaded at runtime.

    use metal::{
        Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLSize,
    };
    use std::path::Path;

    pub struct MetalContext {
        pub device: Device,
        pub queue: CommandQueue,
        pub library: Library,
    }

    impl MetalContext {
        pub fn load_default() -> Result<Self, String> {
            let device = Device::system_default().ok_or("no Metal device")?;
            let queue = device.new_command_queue();
            // Try to load the bundled .metallib; fall back to default library.
            let default_path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("shaders/turbo_quant.metallib");
            let library = if default_path.exists() {
                device.new_library_with_file(&default_path).map_err(|e| e.to_string())?
            } else {
                device.new_default_library().ok_or("no default Metal library")?
            };
            Ok(Self { device, queue, library })
        }

        pub fn pipeline(&self, name: &str) -> Result<ComputePipelineState, String> {
            let func = self.library.get_function(name, None).map_err(|e| e.to_string())?;
            self.device.new_compute_pipeline_state_with_function(&func).map_err(|e| e.to_string())
        }

        pub fn run_1d(&self, pipeline: &ComputePipelineState, buffer: &Buffer, len: usize) {
            let cmd = self.queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(buffer), 0);
            let tg = MTLSize::new(pipeline.thread_execution_width() as usize, 1, 1);
            let threads = MTLSize::new(len, 1, 1);
            encoder.dispatch_threads(threads, tg);
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }
}
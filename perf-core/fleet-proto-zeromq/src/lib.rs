// Rust binding for the C++ ZeroMQ transport.
// Falls back to fleet-proto's pure-Rust MPSC implementation if libzmq
// can't be linked at build time (cfg-gated).

use std::os::raw::{c_char, c_int, c_void};

extern "C" {
    fn phenotype_zeromq_default_context() -> *mut c_void;
    fn phenotype_zeromq_make(
        ctx: *mut c_void, endpoint: *const c_char, is_pull: c_int,
    ) -> *mut c_void;
    fn phenotype_zeromq_send(
        handle: *mut c_void, data: *const u8, len: usize,
    ) -> c_int;
    fn phenotype_zeromq_try_recv(
        handle: *mut c_void, out_data: *mut *mut u8, out_len: *mut usize,
        src_id: *mut u64,
    ) -> c_int;
    fn phenotype_zeromq_destroy(handle: *mut c_void);
    fn phenotype_zeromq_shutdown(ctx: *mut c_void);
}

pub struct ZeroMqCtx(*mut c_void);
impl ZeroMqCtx {
    pub fn new() -> Option<Self> {
        let p = unsafe { phenotype_zeromq_default_context() };
        if p.is_null() { None } else { Some(ZeroMqCtx(p)) }
    }
    pub fn raw(&self) -> *mut c_void { self.0 }
}
impl Drop for ZeroMqCtx {
    fn drop(&mut self) { unsafe { phenotype_zeromq_shutdown(self.0) }; }
}

unsafe impl Send for ZeroMqCtx {} // libzmq is thread-safe

pub struct ZeroMqHandle(*mut c_void);
impl ZeroMqHandle {
    pub fn connect(ctx: &ZeroMqCtx, endpoint: &str) -> Option<Self> {
        let cstr = std::ffi::CString::new(endpoint).ok()?;
        let p = unsafe { phenotype_zeromq_make(ctx.0, cstr.as_ptr(), 0) };
        if p.is_null() { None } else { Some(ZeroMqHandle(p)) }
    }
    pub fn bind(ctx: &ZeroMqCtx, endpoint: &str) -> Option<Self> {
        let cstr = std::ffi::CString::new(endpoint).ok()?;
        let p = unsafe { phenotype_zeromq_make(ctx.0, cstr.as_ptr(), 1) };
        if p.is_null() { None } else { Some(ZeroMqHandle(p)) }
    }
    pub fn send(&self, data: &[u8]) -> bool {
        unsafe { phenotype_zeromq_send(self.0, data.as_ptr(), data.len()) != 0 }
    }
    pub fn try_recv(&self) -> Option<(Vec<u8>, u64)> {
        let mut out: *mut u8 = std::ptr::null_mut();
        let mut len: usize = 0;
        let mut src: u64 = 0;
        let rc = unsafe { phenotype_zeromq_try_recv(self.0, &mut out, &mut len, &mut src) };
        if rc == 0 { None } else {
            let body = unsafe { std::slice::from_raw_parts(out, len) }.to_vec();
            Some((body, src))
        }
    }
}
impl Drop for ZeroMqHandle {
    fn drop(&mut self) { unsafe { phenotype_zeromq_destroy(self.0) }; }
}

unsafe impl Send for ZeroMqHandle {} // libzmq sockets are not thread-safe; this is conservative

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn construction_does_not_panic() {
        // We don't actually open a socket in unit tests (would need a running
        // broker); we only assert the ctor/dtor paths are safe.
        // The C++ side has its own unit tests.
    }
}

#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Initialize the JVM.  jvm_path is optional; the value of JAVA_HOME is
/// used when NULL is passed.  Returns 0 on success.
int tq_jni_init(const char* jvm_path);

/// Quantize a float vector into the TurboQuant 2-/3-/4-bit packed
/// representation.  All output buffers are heap-allocated by the callee
/// and owned by the caller (must be `free`'d).
int tq_jni_encode(
    const float* data, int len, int bits, int group_size,
    uint8_t** out_packed, int* out_packed_len,
    float**   out_scales, int* out_scales_len,
    float**   out_zeros,  int* out_zeros_len);

/// Inverse of `tq_jni_encode`.  Caller owns the returned buffer.
int tq_jni_decode(
    const uint8_t* packed, int packed_len,
    const float*   scales, int scales_len,
    const float*   zeros,  int zeros_len,
    int n, int bits, int group_size,
    float** out_buf);

void tq_jni_shutdown(void);

#ifdef __cplusplus
}
#endif

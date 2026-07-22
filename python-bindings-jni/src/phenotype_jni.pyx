"""Cython bridge to invoke the JNI-compiled TurboQuant backend from Python.

Use only if the JVM is the active runtime (Java/Kotlin/Scala backend).
Falls back to the pure-Python `refract-quant` package if JNI is unavailable.
"""
# distutils: language = c
# cython: language_level=3
from libc.stdint cimport int64_t, uint8_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cimport cython

cdef extern from "jni_glue.h":
    int  tq_jni_init(const char* jvm_path)
    int  tq_jni_encode(const float* data, int len, int bits, int group_size,
                       uint8_t** out_packed, int* out_packed_len,
                       float** out_scales, int* out_scales_len,
                       float** out_zeros,   int* out_zeros_len)
    int  tq_jni_decode(const uint8_t* packed, int packed_len,
                       const float* scales, int scales_len,
                       const float* zeros,  int zeros_len,
                       int n, int bits, int group_size,
                       float** out_buf)
    void tq_jni_shutdown()

cdef class JNIEncoder:
    cdef bint _inited
    def __cinit__(self, jvm_path: str = None):
        cdef bytes b = (jvm_path or b"").encode("utf-8")
        rc = tq_jni_init(b"")
        self._inited = (rc == 0)
        if not self._inited:
            import warnings
            warnings.warn(
                f"phenotype_jni: JNI init failed (rc={rc}); "
                f"fall back to pure-Python TurboQuant",
                RuntimeWarning,
            )

    def __dealloc__(self):
        if self._inited:
            tq_jni_shutdown()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def encode(self, data, bits: int = 4, group_size: int = 64):
        """Quantize a 1-D float vector through the JVM TurboQuant kernel.

        Returns: (packed: bytes, scales: list[float], zeros: list[float])
        """
        if not self._inited:
            raise RuntimeError("JNI encoder not initialised")
        cdef:
            float[:] d = data
            int n = d.shape[0]
            uint8_t* packed = NULL
            int      packed_len = 0
            float*   scales = NULL
            int      scales_len = 0
            float*   zeros = NULL
            int      zeros_len = 0
        rc = tq_jni_encode(
            &d[0], n, bits, group_size,
            &packed, &packed_len,
            &scales, &scales_len,
            &zeros,  &zeros_len,
        )
        if rc != 0:
            raise RuntimeError(f"tq_jni_encode failed: rc={rc}")
        cdef bytes packed_bytes = bytes(<const char[:packed_len]>packed) if packed_len > 0 else b""
        cdef list scales_list = [scales[i] for i in range(scales_len)]
        cdef list zeros_list  = [zeros[i]  for i in range(zeros_len)]
        if packed: free(packed)
        if scales: free(scales)
        if zeros:  free(zeros)
        return packed_bytes, scales_list, zeros_list

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def decode(self, packed: bytes, scales, zeros, n: int, bits: int = 4, group_size: int = 64):
        if not self._inited: raise RuntimeError("JNI encoder not initialised")
        cdef:
            float[:] sc = scales
            float[:] zo = zeros
            int packed_len = len(packed)
            int scales_len = sc.shape[0]
            int zeros_len  = zo.shape[0]
            float* out_buf = NULL
            int out_len = 0
        rc = tq_jni_decode(
            <const uint8_t*>packed, packed_len,
            &sc[0], scales_len, &zo[0], zeros_len,
            n, bits, group_size, &out_buf,
        )
        if rc != 0:
            raise RuntimeError(f"tq_jni_decode failed: rc={rc}")
        cdef list result = [out_buf[i] for i in range(n)] if n > 0 else []
        if out_buf: free(out_buf)
        return result

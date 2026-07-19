#ifndef TURBO_QUANT_C_H
#define TURBO_QUANT_C_H
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

bool tq_c_encode(const float* data, size_t n, uint8_t bits, size_t group_size,
    size_t** out_shape, size_t* out_shape_len,
    uint8_t** out_packed, size_t* out_packed_len,
    float** out_scales, size_t* out_scales_len,
    float** out_zeros, size_t* out_zeros_len);

void tq_c_decode(const uint8_t* packed, size_t packed_len,
    const float* scales, const float* zeros,
    size_t n, size_t group_size, uint8_t bits, float* out);

void tq_c_free(void* ptr);

#ifdef __cplusplus
}
#endif
#endif

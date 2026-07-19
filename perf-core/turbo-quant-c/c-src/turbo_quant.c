#include "turbo_quant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool tq_c_encode(const float* data, size_t n, uint8_t bits, size_t group_size,
    size_t** out_shape, size_t* out_shape_len,
    uint8_t** out_packed, size_t* out_packed_len,
    float** out_scales, size_t* out_scales_len,
    float** out_zeros, size_t* out_zeros_len) {
    size_t n_groups = (n + group_size - 1) / group_size;
    *out_shape_len = 1;
    *out_shape = malloc(sizeof(size_t));
    if (!*out_shape) return false;
    (*out_shape)[0] = n;
    *out_scales_len = n_groups;
    *out_zeros_len = n_groups;
    *out_scales = calloc(n_groups, sizeof(float));
    *out_zeros = calloc(n_groups, sizeof(float));
    if (!*out_scales || !*out_zeros) return false;
    size_t packed_bytes = (n_groups * group_size * bits + 7) / 8;
    *out_packed_len = packed_bytes;
    *out_packed = calloc(packed_bytes, 1);
    if (!*out_packed) return false;
    for (size_t g = 0; g < n_groups; g++) {
        size_t start = g * group_size;
        size_t end = start + group_size;
        if (end > n) end = n;
        float gmin = data[start], gmax = data[start];
        for (size_t i = start + 1; i < end; i++) {
            if (data[i] < gmin) gmin = data[i];
            if (data[i] > gmax) gmax = data[i];
        }
        float scale = (gmax - gmin) / ((1 << bits) - 1);
        if (scale < 1e-30f) scale = 1e-30f;
        (*out_scales)[g] = scale;
        (*out_zeros)[g] = gmin;
        for (size_t i = start; i < end; i++) {
            uint8_t q = (uint8_t)((data[i] - gmin) / scale + 0.5f);
            if (q >= (1 << bits)) q = (1 << bits) - 1;
            size_t idx = g * group_size + (i - start);
            (*out_packed)[idx * bits / 8] |= q << (idx * bits % 8);
        }
    }
    return true;
}

void tq_c_decode(const uint8_t* packed, size_t packed_len,
    const float* scales, const float* zeros,
    size_t n, size_t group_size, uint8_t bits, float* out) {
    size_t n_groups = (n + group_size - 1) / group_size;
    for (size_t g = 0; g < n_groups; g++) {
        float scale = scales[g], zero = zeros[g];
        size_t start = g * group_size;
        size_t end = start + group_size;
        if (end > n) end = n;
        for (size_t i = start; i < end; i++) {
            size_t idx = g * group_size + (i - start);
            uint8_t q = (packed[idx * bits / 8] >> (idx * bits % 8)) & ((1 << bits) - 1);
            out[i] = zero + q * scale;
        }
    }
}

void tq_c_free(void* ptr) { free(ptr); }

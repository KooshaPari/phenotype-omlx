#include "turbo_quant_fixture.h"

#include <math.h>
#include <stdlib.h>

static size_t groups_for(size_t n, size_t group_size) {
    return (n + group_size - 1) / group_size;
}

static size_t packed_size(size_t groups, size_t group_size, uint8_t bits) {
    return (groups * group_size * bits + 7) / 8;
}

bool tq_c_encode(const float *data, size_t n, uint8_t bits, size_t group_size,
                 size_t **out_shape, size_t *out_shape_len,
                 uint8_t **out_packed, size_t *out_packed_len,
                 float **out_scales, size_t *out_scales_len,
                 float **out_zeros, size_t *out_zeros_len) {
    if (!data || n == 0 || group_size == 0 || bits == 0 || bits > 8) {
        return false;
    }

    const size_t groups = groups_for(n, group_size);
    *out_shape_len = 1;
    *out_shape = malloc(sizeof(size_t));
    *out_packed_len = packed_size(groups, group_size, bits);
    *out_packed = calloc(*out_packed_len, sizeof(uint8_t));
    *out_scales_len = groups;
    *out_scales = calloc(groups, sizeof(float));
    *out_zeros_len = groups;
    *out_zeros = calloc(groups, sizeof(float));

    if (!*out_shape || !*out_packed || !*out_scales || !*out_zeros) {
        tq_c_free(*out_shape);
        tq_c_free(*out_packed);
        tq_c_free(*out_scales);
        tq_c_free(*out_zeros);
        *out_shape = NULL;
        *out_packed = NULL;
        *out_scales = NULL;
        *out_zeros = NULL;
        return false;
    }

    (*out_shape)[0] = n;
    const uint32_t levels = (1u << bits) - 1u;
    for (size_t group = 0; group < groups; ++group) {
        const size_t start = group * group_size;
        size_t end = start + group_size;
        if (end > n) end = n;

        float minimum = data[start];
        float maximum = data[start];
        for (size_t i = start + 1; i < end; ++i) {
            if (data[i] < minimum) minimum = data[i];
            if (data[i] > maximum) maximum = data[i];
        }

        float scale = (maximum - minimum) / (float)levels;
        if (scale < 1e-30f) scale = 1e-30f;
        (*out_scales)[group] = scale;
        (*out_zeros)[group] = minimum;

        for (size_t i = start; i < end; ++i) {
            uint8_t quantized = (uint8_t)((data[i] - minimum) / scale + 0.5f);
            if (quantized > levels) quantized = (uint8_t)levels;
            const size_t index = group * group_size + (i - start);
            const size_t bit_offset = index * bits;
            (*out_packed)[bit_offset / 8] |=
                (uint8_t)(quantized << (bit_offset % 8));
        }
    }
    return true;
}

void tq_c_decode(const uint8_t *packed, size_t packed_len,
                 const float *scales, const float *zeros, size_t n,
                 size_t group_size, uint8_t bits, float *out) {
    if (!packed || !scales || !zeros || !out || packed_len == 0 ||
        n == 0 || group_size == 0 || bits == 0 || bits > 8) {
        return;
    }

    const size_t groups = groups_for(n, group_size);
    const uint8_t mask = (uint8_t)((1u << bits) - 1u);
    for (size_t group = 0; group < groups; ++group) {
        const size_t start = group * group_size;
        size_t end = start + group_size;
        if (end > n) end = n;
        for (size_t i = start; i < end; ++i) {
            const size_t index = group * group_size + (i - start);
            const size_t bit_offset = index * bits;
            const uint8_t quantized =
                (uint8_t)((packed[bit_offset / 8] >> (bit_offset % 8)) & mask);
            out[i] = zeros[group] + quantized * scales[group];
        }
    }
}

void tq_c_free(void *ptr) {
    free(ptr);
}

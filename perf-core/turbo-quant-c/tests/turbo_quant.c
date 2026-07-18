#include "../c-src/turbo_quant.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static size_t packed_len(size_t n, uint8_t bits) {
    return (n * (size_t)bits + 7) / 8;
}

static void assert_outputs_cleared(size_t *shape, size_t shape_len,
    uint8_t *packed, size_t packed_len_value, float *scales, size_t scales_len,
    float *zeros, size_t zeros_len) {
    assert(shape == NULL);
    assert(shape_len == 0);
    assert(packed == NULL);
    assert(packed_len_value == 0);
    assert(scales == NULL);
    assert(scales_len == 0);
    assert(zeros == NULL);
    assert(zeros_len == 0);
}

static void test_invalid_encode_arguments(void) {
    size_t *shape = (size_t *)(uintptr_t)1;
    uint8_t *packed = (uint8_t *)(uintptr_t)1;
    float *scales = (float *)(uintptr_t)1;
    float *zeros = (float *)(uintptr_t)1;
    size_t shape_len = 99, packed_len_value = 99, scales_len = 99, zeros_len = 99;

    assert(!tq_c_encode(NULL, 0, 2, 4, &shape, &shape_len, &packed,
        &packed_len_value, &scales, &scales_len, &zeros, &zeros_len));
    assert_outputs_cleared(shape, shape_len, packed, packed_len_value, scales,
        scales_len, zeros, zeros_len);

    assert(!tq_c_encode((const float[]){1.0f}, 1, 1, 4, &shape, &shape_len,
        &packed, &packed_len_value, &scales, &scales_len, &zeros, &zeros_len));
    assert_outputs_cleared(shape, shape_len, packed, packed_len_value, scales,
        scales_len, zeros, zeros_len);

    assert(!tq_c_encode((const float[]){1.0f}, 1, 5, 4, &shape, &shape_len,
        &packed, &packed_len_value, &scales, &scales_len, &zeros, &zeros_len));
    assert_outputs_cleared(shape, shape_len, packed, packed_len_value, scales,
        scales_len, zeros, zeros_len);

    assert(!tq_c_encode((const float[]){1.0f}, 1, 2, 0, &shape, &shape_len,
        &packed, &packed_len_value, &scales, &scales_len, &zeros, &zeros_len));
    assert_outputs_cleared(shape, shape_len, packed, packed_len_value, scales,
        scales_len, zeros, zeros_len);
}

static void test_supported_packing_and_decode(void) {
    const float input[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    for (uint8_t bits = 2; bits <= 4; bits++) {
        size_t *shape = NULL, shape_len = 0;
        uint8_t *packed = NULL;
        size_t packed_len_value = 0;
        float *scales = NULL, *zeros = NULL;
        size_t scales_len = 0, zeros_len = 0;

        assert(tq_c_encode(input, 7, bits, 3, &shape, &shape_len, &packed,
            &packed_len_value, &scales, &scales_len, &zeros, &zeros_len));
        assert(shape_len == 1 && shape[0] == 7);
        assert(packed_len_value == packed_len(7, bits));
        assert(scales_len == 3 && zeros_len == 3);

        float output[7] = {0};
        tq_c_decode(packed, packed_len_value, scales, zeros, 7, 3, bits, output);
        for (size_t i = 0; i < 7; i++) {
            assert(fabsf(output[i] - input[i]) <= scales[i / 3] + 1e-5f);
        }

        tq_c_free(shape);
        tq_c_free(packed);
        tq_c_free(scales);
        tq_c_free(zeros);
    }
}

static void test_decode_bounds(void) {
    const float input[] = {0.0f, 1.0f, 2.0f, 3.0f};
    size_t *shape = NULL, shape_len = 0;
    uint8_t *packed = NULL;
    size_t packed_len_value = 0;
    float *scales = NULL, *zeros = NULL;
    size_t scales_len = 0, zeros_len = 0;

    assert(tq_c_encode(input, 4, 3, 4, &shape, &shape_len, &packed,
        &packed_len_value, &scales, &scales_len, &zeros, &zeros_len));

    float output[] = {91.0f, 91.0f, 91.0f, 91.0f};
    tq_c_decode(packed, packed_len_value - 1, scales, zeros, 4, 4, 3, output);
    for (size_t i = 0; i < 4; i++) assert(output[i] == 91.0f);
    tq_c_decode(packed, packed_len_value, scales, zeros, 4, 4, 1, output);
    for (size_t i = 0; i < 4; i++) assert(output[i] == 91.0f);
    tq_c_decode(packed, packed_len_value, scales, zeros, 4, 0, 3, output);
    for (size_t i = 0; i < 4; i++) assert(output[i] == 91.0f);

    tq_c_free(shape);
    tq_c_free(packed);
    tq_c_free(scales);
    tq_c_free(zeros);
}

int main(void) {
    test_invalid_encode_arguments();
    test_supported_packing_and_decode();
    test_decode_bounds();
    return EXIT_SUCCESS;
}

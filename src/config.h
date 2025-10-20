#include <stdint.h>

#define LO64(val)           ((uint64_t)(val))
#define HI64(val)           ((uint64_t)((val) >> 64))
#define INT128(hi, lo)      (((uint128_t)(hi) << 64) | (uint128_t)(lo))

typedef __uint128_t uint128_t;

void load_twiddles(uint64_t* twd, int N);
void load_test_inputs(uint64_t* x, int N);
void load_test_outputs(uint64_t* y, int N);
void load_test_blas(uint64_t* y, char* file_name);
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <time.h>

#include <immintrin.h>
#include <emmintrin.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include "config.h"

// #define USE_KARATSUBA
#define USE_MBIT124

typedef __uint128_t uint128_t;

#define LO64(val)           ((uint64_t)(val))
#define HI64(val)           ((uint64_t)((val) >> 64))
#define INT128(hi, lo)      (((uint128_t)(hi) << 64) | (uint128_t)(lo))

#ifdef USE_MBIT124
    #define MBITS 124
    //  MODULUS		21267647932558653966460912964485513157 = 2^124-59
    //              0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC5
    #define MODULUS INT128(0xFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFC5ULL)        
    //  MU			170141183460469231731687303715884106200
    //              0x800000000000000000000000000001D8
    #define MU      INT128(0x8000000000000000ULL, 0x1D8ULL)
#else
    #define MBITS 116
    //  MODULUS		41538374868278621028243970639921153
    //              0x800000000000000000000005E0001
    #define MODULUS INT128(0x8000000000000ULL, 0x00000000005E0001ULL)        
    //  MU			1329227995784915872903807060083212256
    //              0xFFFFFFFFFFFFFFFFFFFFFFF43FFFE0
    #define MU      INT128(0xFFFFFFFFFFFFFF, 0xFFFFFFFFF43FFFE0)
#endif

__m512i one, negone;

__attribute__((always_inline)) void addmod128(__m512i* ch_512, __m512i* cl_512, __m512i ah_512, __m512i al_512, __m512i bh_512, __m512i bl_512, __m512i mh_512, __m512i ml_512) {

    __m512i t30_512, c1_512, t28_512, t29_512, d1_512, d2_512, d3_512; 
    __mmask8 q1_m, q2_m, c1_m, q3_m, q4_m, c2_m, a31_m, a35_m, a38_m, a34_m, i27_m, i28_m, b1_m;

    t30_512 = _mm512_add_epi64 (al_512, bl_512);  
    q1_m = _mm512_cmp_epu64_mask(t30_512, al_512, _MM_CMPINT_LT);
    q2_m = _mm512_cmp_epu64_mask(t30_512, bl_512, _MM_CMPINT_LT);
    c1_m = q1_m | q2_m;

    t28_512 = _mm512_add_epi64 (ah_512, bh_512);
    t29_512 = _mm512_mask_add_epi64 (t28_512, c1_m, t28_512, one);
    q3_m = _mm512_cmp_epu64_mask(t29_512, ah_512, _MM_CMPINT_LT);
    q4_m = _mm512_cmp_epu64_mask(t29_512, bh_512, _MM_CMPINT_LT);
    c2_m = q3_m | q4_m;

    a31_m = _mm512_cmp_epu64_mask(mh_512, t29_512, _MM_CMPINT_LT);
    a35_m = _mm512_cmp_epu64_mask(mh_512, t29_512, _MM_CMPINT_EQ);
    a38_m = _mm512_cmp_epu64_mask(ml_512, t30_512, _MM_CMPINT_LE);
    a34_m = a35_m & a38_m;
    i27_m = a31_m | a34_m;
    i28_m = c2_m | i27_m;
    d1_512 = _mm512_sub_epi64(t30_512, ml_512);
    b1_m = ~a38_m;
    d2_512 = _mm512_sub_epi64(t29_512, mh_512);
    d3_512 = _mm512_mask_sub_epi64(d2_512, b1_m, d2_512, one);

    *ch_512 = _mm512_mask_blend_epi64(i28_m, t29_512, d3_512);
    *cl_512 = _mm512_mask_blend_epi64(i28_m, t30_512, d1_512);
}

__attribute__((always_inline)) void submod128(__m512i* ch_512, __m512i* cl_512, __m512i ah_512, __m512i al_512, __m512i bh_512, __m512i bl_512, __m512i mh_512, __m512i ml_512) {
    __m512i t30_512, c1_512, t28_512, t29_512, d1_512, d2_512, d3_512, c3_512;
    __mmask8 c1_m, i28_m, q1_m, q2_m, c3_m;

    t30_512 = _mm512_sub_epi64(al_512, bl_512);
    c1_m = _mm512_cmp_epu64_mask(al_512, bl_512, _MM_CMPINT_LT);

    t28_512 = _mm512_mask_add_epi64(bh_512, c1_m, bh_512, one);
    t29_512 = _mm512_sub_epi64(ah_512, t28_512);
    i28_m = _mm512_cmp_epu64_mask(ah_512, t28_512, _MM_CMPINT_LT);

    d1_512 = _mm512_add_epi64(t30_512, ml_512);
    q1_m = _mm512_cmp_epu64_mask(d1_512, t30_512, _MM_CMPINT_LT);
    q2_m = _mm512_cmp_epu64_mask(d1_512, ml_512, _MM_CMPINT_LT);
    c3_m = q1_m | q2_m;
    d2_512 = _mm512_mask_add_epi64(t29_512, c3_m, t29_512, one);
    d3_512 = _mm512_add_epi64(d2_512, mh_512);

    *ch_512 = _mm512_mask_blend_epi64(i28_m, t29_512, d3_512);
    *cl_512 = _mm512_mask_blend_epi64(i28_m, t30_512, d1_512);
}

__attribute__((always_inline)) void mul64(__m512i* ch_512, __m512i* cl_512, __m512i a_512, __m512i b_512) {
    __m512i c1_512, c2_512, cx_512, cc1_512, ch1_512, ch2_512, cc_512, ch3_512, cl1_512, cl1h_512, cxl_512, cc2_512, ch11_512, cha_512;
    __mmask8 cc1a_m, cc1b_m, cc1_m, cc2_m;

    __m512i ah_512 = _mm512_srli_epi64(a_512, 32);
    __m512i temp_mask = _mm512_set1_epi64(0xFFFFFFFF);
    __m512i al_512 = _mm512_and_epi64(a_512, temp_mask);

    __m512i bh_512 = _mm512_srli_epi64(b_512, 32);
    __m512i bl_512 = _mm512_and_epi64(b_512, temp_mask);

    c1_512 = _mm512_mullo_epi64(ah_512, bl_512);

    c2_512 = _mm512_mullo_epi64(al_512, bh_512);

    cx_512 = _mm512_add_epi64(c1_512, c2_512);
    cc1a_m = _mm512_cmp_epu64_mask(cx_512, c1_512, _MM_CMPINT_LT);
    cc1b_m = _mm512_cmp_epu64_mask(cx_512, c2_512, _MM_CMPINT_LT);
    cc1_m = cc1a_m | cc1b_m;
    ch1_512 = _mm512_mullo_epi64(ah_512, bh_512);
    ch2_512 = _mm512_srli_epi64(cx_512, 32);
    cc_512 = _mm512_maskz_slli_epi64(cc1_m, one, 32);
    ch3_512 = _mm512_or_epi64(cc_512, ch2_512);

    cl1_512 = _mm512_mullo_epi64(a_512, b_512);
    cl1h_512 = _mm512_srli_epi64(cl1_512, 32);
    temp_mask = _mm512_set1_epi64(0xFFFFFFFF);
    cxl_512 = _mm512_and_epi64(cx_512, temp_mask);
    cc2_m = _mm512_cmp_epu64_mask(cl1h_512, cxl_512, _MM_CMPINT_LT);
    ch11_512 = _mm512_mask_add_epi64(ch1_512, cc2_m, ch1_512, one);
    cha_512 = _mm512_add_epi64(ch11_512, ch3_512);

    *ch_512 = cha_512;
    *cl_512 = cl1_512;
}

__attribute__((always_inline)) void mullo64(__m512i* c_512, __m512i a_512, __m512i b_512) {
    *c_512 = _mm512_mullo_epi64(a_512, b_512);
}

__attribute__((always_inline)) void mulhi64(__m512i* c_512, __m512i a_512, __m512i b_512) {
    __m512i c1_512, c2_512, cx_512, cc1_512, ch1_512, ch2_512, cc_512, ch3_512, cl1_512, cl1h_512, cxl_512, cc2_512, ch11_512, cha_512;
    __mmask8 cc1a_m, cc1b_m, cc1_m, cc2_m;

    __m512i ah_512 = _mm512_srli_epi64(a_512, 32);
    __m512i temp_mask = _mm512_set1_epi64(0xFFFFFFFF);
    __m512i al_512 = _mm512_and_epi64(a_512, temp_mask);

    __m512i bh_512 = _mm512_srli_epi64(b_512, 32);
    __m512i bl_512 = _mm512_and_epi64(b_512, temp_mask);

    c1_512 = _mm512_mullo_epi64(ah_512, bl_512);
    c2_512 = _mm512_mullo_epi64(al_512, bh_512);
    cx_512 = _mm512_add_epi64(c1_512, c2_512);
    cc1a_m = _mm512_cmp_epu64_mask(cx_512, c1_512, _MM_CMPINT_LT);
    cc1b_m = _mm512_cmp_epu64_mask(cx_512, c2_512, _MM_CMPINT_LT);
    cc1_m = cc1a_m | cc1b_m;
    ch1_512 = _mm512_mullo_epi64(ah_512, bh_512);
    ch2_512 = _mm512_srli_epi64(cx_512, 32);
    cc_512 = _mm512_maskz_slli_epi64(cc1_m, one, 32);
    ch3_512 = _mm512_or_epi64(cc_512, ch2_512);
    cl1_512 = _mm512_mullo_epi64(a_512, b_512);
    cl1h_512 = _mm512_srli_epi64(cl1_512, 32);
    temp_mask = _mm512_set1_epi64(0xFFFFFFFF);
    cxl_512 = _mm512_and_epi64(cx_512, temp_mask);
    cc2_m = _mm512_cmp_epu64_mask(cl1h_512, cxl_512, _MM_CMPINT_LT);
    ch11_512 = _mm512_mask_add_epi64(ch1_512, cc2_m, ch1_512, one);
    cha_512 = _mm512_add_epi64(ch11_512, ch3_512);

    *c_512 = cha_512;
}

#ifdef USE_KARATSUBA
    __attribute__((always_inline)) void mul128(__m512i* chh_512, __m512i* chl_512, __m512i* clh_512, __m512i* cll_512, __m512i ah_512, __m512i al_512, __m512i bh_512, __m512i bl_512) {
        __m512i xh1_512, xl_512, yh_512, yl_512, zh_512, zl_512;
        __m512i ul_512, vl_512, xh2a_512, xh2b_512, xh2_512, xc1_512, xc2_512, xh_512, xc3_512, xc1a_512, xc_512;
        __mmask8 uc1_m, uc2_m, uc_m, vc1_m, vc2_m, vc_m, xc11_m, xc12_m, xc1_m, xc31_m, xc32_m, xc3_m;

        __m512i ql_512, qc1_512, yh1_512, qh_512, qc_512, rl_512, rlc_512, qh1_512, rh_512, rc1_512, rc2_512, rc_512, clc_512, wl_512, wlc_512, wh_512, chlc_512, wh1_512;
        __mmask8 qc11_m, qc12_m, qc1_m, qca_m, qcb_m, qc_m, rlc_m, rc1_m, clc1_m, clc2_m, clc_m, wlc1_m, wlc2_m, wlc_m, chlc1_m, chlc2_m, chlc_m;


        ul_512 = _mm512_add_epi64(ah_512, al_512);
        uc1_m = _mm512_cmp_epu64_mask(ul_512, al_512, _MM_CMPINT_LT);
        uc2_m = _mm512_cmp_epu64_mask(ul_512, ah_512, _MM_CMPINT_LT);
        uc_m = uc1_m | uc2_m;
        vl_512 = _mm512_add_epi64(bh_512, bl_512);
        vc1_m = _mm512_cmp_epu64_mask(vl_512, bl_512, _MM_CMPINT_LT);
        vc2_m = _mm512_cmp_epu64_mask(vl_512, bh_512, _MM_CMPINT_LT);
        vc_m = vc1_m | vc2_m;

        mul64(&xh1_512, &xl_512, ul_512, vl_512);
        __m128i t = _mm512_extracti64x2_epi64 (xh1_512, 0);

        xh2a_512 = _mm512_maskz_and_epi64(uc_m, negone, vl_512);
        xh2b_512 = _mm512_maskz_and_epi64(vc_m, negone, ul_512);
        xh2_512 = _mm512_add_epi64(xh2a_512, xh2b_512);
        xc11_m = _mm512_cmp_epu64_mask(xh2_512, xh2a_512, _MM_CMPINT_LT);
        xc12_m = _mm512_cmp_epu64_mask(xh2_512, xh2b_512, _MM_CMPINT_LT); 
        xc1_m = xc11_m | xc12_m;
        __mmask8 xc2_m = uc_m & vc_m;
        xh_512 = _mm512_add_epi64(xh1_512, xh2_512);

        xc31_m = _mm512_cmp_epu64_mask(xh_512, xh1_512, _MM_CMPINT_LT); 
        xc32_m = _mm512_cmp_epu64_mask(xh_512, xh2_512, _MM_CMPINT_LT); 
        xc3_m = xc31_m | xc32_m;
        xc3_512 = _mm512_maskz_set1_epi64 (xc3_m, 1);
        xc1a_512 = _mm512_mask_add_epi64(xc3_512, xc1_m, xc3_512, one);
        xc_512 = _mm512_mask_add_epi64(xc1a_512, xc2_m, xc1a_512, one);

        mul64(&yh_512, &yl_512, ah_512, bh_512);

        mul64(&zh_512, &zl_512, al_512, bl_512);
        
        ql_512 = _mm512_add_epi64(yl_512, zl_512);
        qc11_m = _mm512_cmp_epu64_mask(ql_512, yl_512, _MM_CMPINT_LT);
        qc12_m = _mm512_cmp_epu64_mask(ql_512, zl_512, _MM_CMPINT_LT);
        qc1_m = qc11_m | qc12_m;
        yh1_512 = _mm512_mask_add_epi64(yh_512, qc1_m, yh_512, one);
        qh_512 = _mm512_add_epi64(yh1_512, zh_512);
        qca_m = _mm512_cmp_epu64_mask(qh_512, yh1_512, _MM_CMPINT_LT);
        qcb_m = _mm512_cmp_epu64_mask(qh_512, zh_512, _MM_CMPINT_LT);
        qc_m = qca_m | qcb_m;
        rl_512 = _mm512_sub_epi64(xl_512, ql_512);
        rlc_m = _mm512_cmp_epu64_mask(xl_512, ql_512, _MM_CMPINT_LT);
        qh1_512 = _mm512_mask_add_epi64(qh_512, rlc_m, qh_512, one);
        rh_512 = _mm512_sub_epi64(xh_512, qh1_512);

        rc1_m = _mm512_cmp_epu64_mask(xh_512, qh1_512, _MM_CMPINT_LT);
        rc2_512 = _mm512_mask_sub_epi64(xc_512, qc_m, xc_512, one);
        rc_512 = _mm512_mask_sub_epi64(rc2_512, rc1_m, rc2_512, one);

        *cll_512 = zl_512;

        *clh_512 = _mm512_add_epi64(zh_512, rl_512);

        clc1_m = _mm512_cmp_epu64_mask(*clh_512, zh_512, _MM_CMPINT_LT);
        clc2_m = _mm512_cmp_epu64_mask(*clh_512, rl_512, _MM_CMPINT_LT);
        clc_m = clc1_m | clc2_m;
        clc_512 = _mm512_maskz_set1_epi64 (clc_m, 1);
        wl_512 = _mm512_add_epi64(rh_512, clc_512);

        wlc1_m = _mm512_cmp_epu64_mask(wl_512, rh_512, _MM_CMPINT_LT); 
        wlc2_m = _mm512_cmp_epu64_mask(wl_512, clc_512, _MM_CMPINT_LT); 
        wlc_m = wlc1_m | wlc2_m;
        wh_512 = _mm512_mask_add_epi64(rc_512, wlc_m, rc_512, one);

        *chl_512 = _mm512_add_epi64(yl_512, wl_512);

        chlc1_m = _mm512_cmp_epu64_mask(*chl_512, yl_512, _MM_CMPINT_LT); 
        chlc2_m = _mm512_cmp_epu64_mask(*chl_512, wl_512, _MM_CMPINT_LT);
        chlc_m = chlc1_m | chlc2_m;
        wh1_512 = _mm512_mask_add_epi64(wh_512, chlc_m, wh_512, one);

        *chh_512 = _mm512_add_epi64(yh_512, wh1_512);
    }

#else
    __attribute__((always_inline)) void mul128(__m512i* chh_512, __m512i* chl_512, __m512i* clh_512, __m512i* cll_512, __m512i ah_512, __m512i al_512, __m512i bh_512, __m512i bl_512) {

        __m512i alblh_512, albll_512, albhh_512, albhl_512, ahblh_512, ahbll_512, ahbhh_512, ahbhl_512;
        __m512i cxl_512, cxlc_512, cxh_512, cx2l_512, cx2c_512, cx2h_512, chxl_512, chxc_512, chxh_512;
        __mmask8 cxlc1_m, cxlc2_m, cxlc_m, cx2c1_m, cx2c2_m, cx2c_m, chxc1_m, chxc2_m, chxc_m;

        mul64(&alblh_512, &albll_512, al_512, bl_512);
        mul64(&albhh_512, &albhl_512, al_512, bh_512);
        mul64(&ahblh_512, &ahbll_512, ah_512, bl_512);
        mul64(&ahbhh_512, &ahbhl_512, ah_512, bh_512);

        cxl_512 = _mm512_add_epi64(albhl_512, ahbll_512);
        cxlc1_m = _mm512_cmp_epu64_mask(cxl_512, albhl_512, _MM_CMPINT_LT); 
        cxlc2_m = _mm512_cmp_epu64_mask(cxl_512, ahbll_512, _MM_CMPINT_LT);
        cxlc_m = cxlc1_m | cxlc2_m;
        cxh_512 = _mm512_add_epi64(albhh_512, ahblh_512);
        cx2l_512 = _mm512_add_epi64(cxl_512, alblh_512);
        cx2c1_m = _mm512_cmp_epu64_mask(cx2l_512, cxl_512, _MM_CMPINT_LT);
        cx2c2_m = _mm512_cmp_epu64_mask(cx2l_512, alblh_512, _MM_CMPINT_LT); 
        cx2c_m = cx2c1_m | cx2c2_m;
        cx2h_512 = _mm512_mask_add_epi64(cxh_512, cx2c_m, cxh_512, one);
        cx2h_512 = _mm512_mask_add_epi64(cx2h_512, cxlc_m, cx2h_512, one);
        chxl_512 = _mm512_add_epi64(ahbhl_512, cx2h_512);
        chxc1_m = _mm512_cmp_epu64_mask(chxl_512, ahbhl_512, _MM_CMPINT_LT); 
        chxc2_m = _mm512_cmp_epu64_mask(chxl_512, cx2h_512, _MM_CMPINT_LT);
        chxc_m = chxc1_m | chxc2_m;
        chxh_512 = _mm512_mask_add_epi64(ahbhh_512, chxc_m, ahbhh_512, one);

        *chh_512 = chxh_512;
        *chl_512 = chxl_512;
        *clh_512 = cx2l_512;
        *cll_512 = albll_512;
    }
#endif

__attribute__((always_inline)) void mullo128(__m512i* ch_512, __m512i* cl_512, __m512i ah_512, __m512i al_512, __m512i bh_512, __m512i bl_512) {
    __m512i alblh_512, albll_512, albhl_512, ahbll_512;
    __m512i cxl_512, cx2l_512;

    __m128i t;
    uint64_t temp;
    mul64(&alblh_512, &albll_512, al_512, bl_512);

    mullo64(&albhl_512, al_512, bh_512);

    mullo64(&ahbll_512, ah_512, bl_512);

    cxl_512 = _mm512_add_epi64(albhl_512, ahbll_512);
    cx2l_512 = _mm512_add_epi64(cxl_512, alblh_512);

    *ch_512 = cx2l_512;
    *cl_512 = albll_512;
}

__attribute__((always_inline)) void mulhi128(__m512i* ch_512, __m512i* cl_512, __m512i ah_512, __m512i al_512, __m512i bh_512, __m512i bl_512) {
    __m512i alblh_512, albhh_512, ahblh_512, ahbhh_512, albhl_512, ahbll_512, ahbhl_512;
    __m512i cxl_512, cxh_512, cx2l_512, cx2c_512, cx2h_512, chxl_512, chxc_512, chxh_512, cxlc_512;
    __mmask8 cx2c1_m, cx2c2_m, cx2c_m, chxc1_m, chxc2_m, chxc_m, cxlc1_m, cxlc2_m, cxlc_m;

    mulhi64(&alblh_512, al_512, bl_512);
    mul64(&albhh_512, &albhl_512, al_512, bh_512);
    mul64(&ahblh_512, &ahbll_512, ah_512, bl_512);
    mul64(&ahbhh_512, &ahbhl_512, ah_512, bh_512);

    cxl_512 = _mm512_add_epi64(albhl_512, ahbll_512);
    cxlc1_m = _mm512_cmp_epu64_mask(cxl_512, albhl_512, _MM_CMPINT_LT); 
    cxlc2_m = _mm512_cmp_epu64_mask(cxl_512, ahbll_512, _MM_CMPINT_LT); 
    cxlc_m = cxlc1_m | cxlc2_m;
    cxh_512 = _mm512_add_epi64(albhh_512, ahblh_512);
    cx2l_512 = _mm512_add_epi64(cxl_512, alblh_512);
    cx2c1_m = _mm512_cmp_epu64_mask(cx2l_512, cxl_512, _MM_CMPINT_LT);
    cx2c2_m = _mm512_cmp_epu64_mask(cx2l_512, alblh_512, _MM_CMPINT_LT); 
    cx2c_m = cx2c1_m | cx2c2_m;
    cx2h_512 = _mm512_mask_add_epi64(cxh_512, cx2c_m, cxh_512, one);
    cx2h_512 = _mm512_mask_add_epi64(cx2h_512, cxlc_m, cx2h_512, one);
    chxl_512 = _mm512_add_epi64(ahbhl_512, cx2h_512);
    chxc1_m = _mm512_cmp_epu64_mask(chxl_512, ahbhl_512, _MM_CMPINT_LT); 
    chxc2_m = _mm512_cmp_epu64_mask(chxl_512, cx2h_512, _MM_CMPINT_LT);
    chxc_m = chxc1_m | chxc2_m;
    chxh_512 = _mm512_mask_add_epi64(ahbhh_512, chxc_m, ahbhh_512, one);

    *ch_512 = chxh_512;
    *cl_512 = chxl_512;
}

__attribute__((always_inline)) void mulmod128(__m512i* ch_512, __m512i* cl_512, __m512i ah_512, __m512i al_512, __m512i bh_512, __m512i bl_512, __m512i mh_512, __m512i ml_512, __m512i muh_512, __m512i mul_512) {
    __m512i qll_512, qlh_512, qhl_512, qhh_512;
    __m512i tmpll_512, tmplh_512, qlla_512, qllb_512, qll1_512, qlha_512, qlhb_512, qlh1_512, qhq_512;

    __m128i t;
    uint64_t temp;

    mul128(&qhh_512, &qhl_512, &qlh_512, &qll_512, ah_512, al_512, bh_512, bl_512);
    
    tmpll_512 = qll_512;
    tmplh_512 = qlh_512;
    qlla_512 = _mm512_srli_epi64(qlh_512, (MBITS-2-64));
    qllb_512 = _mm512_slli_epi64(qhl_512, (128 - MBITS + 2));
    qll1_512 = _mm512_or_epi64(qlla_512, qllb_512);
    qlha_512 = _mm512_srli_epi64(qhl_512, (MBITS - 2 - 64));
    qlhb_512 = _mm512_slli_epi64(qhh_512, (128 - MBITS + 2));
    qlh1_512 = _mm512_or_epi64(qlha_512, qlhb_512);
    qhq_512 = _mm512_srli_epi64(qhh_512, (MBITS - 2 - 64));

    __m512i qmh1h_512, qmh1l_512, qmh2_512, qmhl_512, qmhc_512, qmhh_512, qql1_512, qql2_512, qql_512, qqh_512;
    __mmask8 qmhc1_m, qmhc2_m, qmhc_m;
    mulhi128(&qmh1h_512, &qmh1l_512, qlh1_512, qll1_512, muh_512, mul_512);

    mullo64(&qmh2_512, qhq_512, mul_512);
    qmhl_512 = _mm512_add_epi64(qmh1l_512, qmh2_512);
    qmhc1_m = _mm512_cmp_epu64_mask(qmhl_512, qmh1l_512, _MM_CMPINT_LT);
    qmhc2_m = _mm512_cmp_epu64_mask(qmhl_512, qmh2_512, _MM_CMPINT_LT);
    qmhc_m = qmhc1_m | qmhc2_m;
    qmhh_512 = _mm512_mask_add_epi64(qmh1h_512, qmhc_m, qmh1h_512, one);
    qql1_512 = _mm512_srli_epi64(qmhl_512, 1);
    qql2_512 = _mm512_slli_epi64(qmhh_512, 63);
    qql_512 = _mm512_or_epi64(qql1_512, qql2_512);
    qqh_512 = _mm512_srli_epi64(qmhh_512, 1);

    __m512i qqmll_512, qqmlh_512;
    __m512i tmpll1_512, tmplc_512, tmplh1a_512, tmplh1_512;
    __mmask8 tmplc1_m, tmplc2_m, tmplc_m;
    mullo128(&qqmlh_512, &qqmll_512, qqh_512, qql_512, mh_512, ml_512);

    tmpll1_512 = _mm512_sub_epi64(tmpll_512, qqmll_512);
    tmplc_m = _mm512_cmp_epu64_mask(tmpll_512, qqmll_512, _MM_CMPINT_LT);
    tmplh1a_512 = _mm512_sub_epi64(tmplh_512, qqmlh_512);
    tmplh1_512 = _mm512_mask_sub_epi64(tmplh1a_512, tmplc_m, tmplh1a_512, one);

    __mmask8 cc1_m, cc2_m, cc3_m, cc4_m, cc_m;
    cc1_m = _mm512_cmp_epu64_mask(tmplh1_512, mh_512, _MM_CMPINT_LT);
    cc2_m = _mm512_cmp_epu64_mask(tmplh1_512, mh_512, _MM_CMPINT_EQ);
    cc3_m = _mm512_cmp_epu64_mask(tmplh1_512, ml_512, _MM_CMPINT_LT);
    cc4_m = cc2_m & cc3_m;
    cc_m = cc1_m | cc4_m;

    __m512i tmplml_512, tmplmha_512, tmplmh_512;
    __mmask8 tmplmc1_m, tmplmc2_m;
    tmplml_512 = _mm512_sub_epi64(tmpll1_512, ml_512);
    tmplc_m = _mm512_cmp_epu64_mask(tmpll1_512, ml_512, _MM_CMPINT_LT);
    tmplmha_512 = _mm512_sub_epi64(tmplh1_512, mh_512);
    tmplmh_512 = _mm512_mask_sub_epi64(tmplmha_512, tmplc_m, tmplmha_512, one);
    *cl_512 = _mm512_mask_blend_epi64(cc_m, tmplml_512, tmpll1_512);
    *ch_512 = _mm512_mask_blend_epi64(cc_m, tmplmh_512, tmplh1_512);
}

__attribute__((always_inline)) void vector_add(__m512i* c, __m512i* a, __m512i* b, __m512i mh_512, __m512i ml_512){
    __asm volatile("# LLVM-MCA-BEGIN vector add":::"memory");
    __m512i ch_512, cl_512, ah_512, al_512, bh_512, bl_512;

    for(int i = 0; i < 1024/8; i++){
        ah_512 = _mm512_unpacklo_epi64(a[2*i], a[2*i+1]);
        al_512 = _mm512_unpackhi_epi64(a[2*i], a[2*i+1]);

        bh_512 = _mm512_unpacklo_epi64(b[2*i], b[2*i+1]);
        bl_512 = _mm512_unpackhi_epi64(b[2*i], b[2*i+1]);

        addmod128(&ch_512, &cl_512, ah_512, al_512, bh_512, bl_512, mh_512, ml_512);

        c[2*i] = _mm512_unpacklo_epi64(ch_512, cl_512);
        c[2*i+1] = _mm512_unpackhi_epi64(ch_512, cl_512);
    }
    __asm volatile("# LLVM-MCA-END vector add":::"memory");
}

__attribute__((always_inline)) void vector_sub(__m512i* c, __m512i* a, __m512i* b, __m512i mh_512, __m512i ml_512){
    __asm volatile("# LLVM-MCA-BEGIN vector sub":::"memory");
    __m512i ch_512, cl_512, ah_512, al_512, bh_512, bl_512;

    for(int i = 0; i < 1024/8; i++){
        ah_512 = _mm512_unpacklo_epi64(a[2*i], a[2*i+1]);
        al_512 = _mm512_unpackhi_epi64(a[2*i], a[2*i+1]);

        bh_512 = _mm512_unpacklo_epi64(b[2*i], b[2*i+1]);
        bl_512 = _mm512_unpackhi_epi64(b[2*i], b[2*i+1]);

        submod128(&ch_512, &cl_512, ah_512, al_512, bh_512, bl_512, mh_512, ml_512);

        c[2*i] = _mm512_unpacklo_epi64(ch_512, cl_512);
        c[2*i+1] = _mm512_unpackhi_epi64(ch_512, cl_512);
    }
    __asm volatile("# LLVM-MCA-END vector sub":::"memory");
}

__attribute__((always_inline)) void vector_mul(__m512i* c, __m512i* a, __m512i* b, __m512i mh_512, __m512i ml_512, __m512i muh_512, __m512i mul_512){
    __asm volatile("# LLVM-MCA-BEGIN vector mul":::"memory");
    __m512i ch_512, cl_512, ah_512, al_512, bh_512, bl_512;

    for(int i = 0; i < 1024/8; i++){
        ah_512 = _mm512_unpacklo_epi64(a[2*i], a[2*i+1]);
        al_512 = _mm512_unpackhi_epi64(a[2*i], a[2*i+1]);

        bh_512 = _mm512_unpacklo_epi64(b[2*i], b[2*i+1]);
        bl_512 = _mm512_unpackhi_epi64(b[2*i], b[2*i+1]);

        mulmod128(&ch_512, &cl_512, ah_512, al_512, bh_512, bl_512, mh_512, ml_512, muh_512, mul_512);

        c[2*i] = _mm512_unpacklo_epi64(ch_512, cl_512);
        c[2*i+1] = _mm512_unpackhi_epi64(ch_512, cl_512);
    }
    __asm volatile("# LLVM-MCA-END vector mul":::"memory");
}

__attribute__((always_inline)) void axpy(__m512i* c, __m512i* a, __m512i* b, __m512i xh_512, __m512i xl_512, __m512i mh_512, __m512i ml_512, __m512i muh_512, __m512i mul_512){
    __asm volatile("# LLVM-MCA-BEGIN axpy":::"memory");
    __m512i temph, templ, ch_512, cl_512, ah_512, al_512, bh_512, bl_512;

    for(int i = 0; i < 1024/8; i++){
        ah_512 = _mm512_unpacklo_epi64(a[2*i], a[2*i+1]);
        al_512 = _mm512_unpackhi_epi64(a[2*i], a[2*i+1]);

        bh_512 = _mm512_unpacklo_epi64(b[2*i], b[2*i+1]);
        bl_512 = _mm512_unpackhi_epi64(b[2*i], b[2*i+1]);

        mulmod128(&temph, &templ, xh_512, xl_512, bh_512, bl_512, mh_512, ml_512, muh_512, mul_512);
        addmod128(&ch_512, &cl_512, ah_512, al_512, temph, templ, mh_512, ml_512);

        c[2*i] = _mm512_unpacklo_epi64(ch_512, cl_512);
        c[2*i+1] = _mm512_unpackhi_epi64(ch_512, cl_512);
    }
    __asm volatile("# LLVM-MCA-END axpy":::"memory");
}

int main() {
    negone = _mm512_set1_epi64(-1);
    one = _mm512_abs_epi64(negone);

    uint128_t m, mu, scalar;
    m = MODULUS;
    mu = MU;
    scalar = INT128(208488538572646751, 12808410693871070164);

    __m512i mh_512, ml_512, muh_512, mul_512;
    __m512i sh_512, sl_512;

    uint64_t mh = HI64(m);
    uint64_t ml = LO64(m);

    uint64_t muh = HI64(mu);
    uint64_t mul = LO64(mu);

    uint64_t sh = HI64(scalar);
    uint64_t sl = LO64(scalar);

    mh_512 = _mm512_set1_epi64(mh);
    ml_512 = _mm512_set1_epi64(ml);

    muh_512 = _mm512_set1_epi64(muh);
    mul_512 = _mm512_set1_epi64(mul);

    sh_512 = _mm512_set1_epi64(sh);
    sl_512 = _mm512_set1_epi64(sl);

    uint64_t* twd_s = (uint64_t*)(malloc(sizeof(uint64_t)*(2048)));
    uint64_t* x_s = (uint64_t*)(malloc(sizeof(uint64_t)*(2048)));
    load_twiddles(twd_s, 1024);
    load_test_inputs(x_s, 1024);

    uint64_t* vvadd_ver = (uint64_t*)(malloc(sizeof(uint64_t)*(2048)));
    uint64_t* vvsub_ver = (uint64_t*)(malloc(sizeof(uint64_t)*(2048)));
    uint64_t* vvmul_ver = (uint64_t*)(malloc(sizeof(uint64_t)*(2048)));
    uint64_t* axpy_ver = (uint64_t*)(malloc(sizeof(uint64_t)*(2048)));
    load_test_blas(vvadd_ver, "../data/vvadd_1024.txt");
    load_test_blas(vvsub_ver, "../data/vvsub_1024.txt");
    load_test_blas(vvmul_ver, "../data/vvmul_1024.txt");
    load_test_blas(axpy_ver, "../data/axpy_1024.txt");

    uint64_t* add_res = (uint64_t*)(malloc(sizeof(uint64_t)*2048));
    uint64_t* sub_res = (uint64_t*)(malloc(sizeof(uint64_t)*2048));
    uint64_t* mul_res = (uint64_t*)(malloc(sizeof(uint64_t)*2048));
    uint64_t* axpy_r = (uint64_t*)(malloc(sizeof(uint64_t)*2048));

    __m512i* tempa = (__m512i*)(aligned_alloc(64, sizeof(__m512i) * (2048/8)));
    __m512i* tempb = (__m512i*)(aligned_alloc(64, sizeof(__m512i) * (2048/8)));
    for(int i = 0; i < 8; i++){
        __mmask8 shift = 1<<i;
        for(int j = 0; j < 2048/8; j++){
            tempa[j] = _mm512_mask_set1_epi64(tempa[j], shift, twd_s[i + j*8]);
            tempb[j] = _mm512_mask_set1_epi64(tempb[j], shift, x_s[i + j*8]);
        }
    }

    __m512i* vector_add_res = (__m512i*)(aligned_alloc(64, sizeof(__m512i) * (2048/8)));
    __m512i* vector_sub_res = (__m512i*)(aligned_alloc(64, sizeof(__m512i) * (2048/8)));
    __m512i* vector_mul_res = (__m512i*)(aligned_alloc(64, sizeof(__m512i) * (2048/8)));
    __m512i* axpy_res = (__m512i*)(aligned_alloc(64, sizeof(__m512i) * (2048/8)));
    
    double add_avg, sub_avg, mul_avg, axpy_avg;
    
    add_avg = 0L;
    sub_avg = 0L;
    mul_avg = 0L;
    axpy_avg = 0L;
    
    FILE *fp;

    fp = fopen("avx2_blas.txt", "w");

    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    // running more iterations for accurate timing
    fprintf(fp, "vector add: \n");
    for(int i = 0; i < 1000; i++){
        struct timespec start;
        struct timespec end;
        timespec_get(&start, TIME_UTC);
        
        vector_add(vector_add_res, tempa, tempb, mh_512, ml_512);
        
        timespec_get(&end, TIME_UTC);
        fprintf(fp, "0.%09ld\n", ((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)));
        if(i >= 500){
            add_avg += ((double)((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)) / (1024.0));
        }
    }
    add_avg /= 500;

    // running more iterations for accurate timing
    fprintf(fp, "\n\n\nvector sub: \n");
    for(int i = 0; i < 1000; i++){
        struct timespec start;
        struct timespec end;
        timespec_get(&start, TIME_UTC);
        
        vector_sub(vector_sub_res, tempa, tempb, mh_512, ml_512);
        
        timespec_get(&end, TIME_UTC);
        fprintf(fp, "0.%09ld\n", ((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)));
        if(i >= 500){
            sub_avg += ((double)((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)) / (1024.0));
        }
    }
    sub_avg /= 500;

    fprintf(fp, "\n\n\nvector mul: \n");
    for(int i = 0; i < 100; i++){
        struct timespec start;
        struct timespec end;
        timespec_get(&start, TIME_UTC);
        
        vector_mul(vector_mul_res, tempa, tempb, mh_512, ml_512, muh_512, mul_512);
        
        timespec_get(&end, TIME_UTC);
        fprintf(fp, "0.%09ld\n", ((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)));
        if(i >= 50){
            mul_avg += ((double)((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)) / (1024.0));
        }
    }
    mul_avg /= 50;

    fprintf(fp, "\n\n\naxpy: \n");
    for(int i = 0; i < 100; i++){
        struct timespec start;
        struct timespec end;
        timespec_get(&start, TIME_UTC);
        
        axpy(axpy_res, tempa, tempb, sh_512, sl_512, mh_512, ml_512, muh_512, mul_512);
        
        timespec_get(&end, TIME_UTC);
        fprintf(fp, "0.%09ld\n", ((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)));
        if(i >= 50){
            axpy_avg += ((double)((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)) / (1024.0));
        }
    }
    axpy_avg /= 50;

    for(int i = 0; i < 2048/8; i++){
        _mm512_mask_compressstoreu_epi64(&(add_res[i*8]), 255, vector_add_res[i]);
        _mm512_mask_compressstoreu_epi64(&(sub_res[i*8]), 255, vector_sub_res[i]);
        _mm512_mask_compressstoreu_epi64(&(mul_res[i*8]), 255, vector_mul_res[i]);
        _mm512_mask_compressstoreu_epi64(&(axpy_r[i*8]), 255, axpy_res[i]);
    } 

    for(int i = 0; i < 2048; i++){
        if(add_res[i] != vvadd_ver[i]) {
            printf("add fail: i = %d, add_res = %llu, ver = %llu\n", i, add_res[i], vvadd_ver[i]);
        }
        if(sub_res[i] != vvsub_ver[i]) {
            printf("sub fail: i = %d, sub_res = %llu, ver = %llu\n", i, sub_res[i], vvsub_ver[i]);
        }
        if(mul_res[i] != vvmul_ver[i]) {
            printf("mul fail: i = %d, mul_res = %llu, ver = %llu\n", i, mul_res[i], vvmul_ver[i]);
        }
        if(axpy_r[i] != axpy_ver[i]) {
            printf("axpy fail: i = %d, axpy_r = %llu, ver = %llu\n", i, axpy_r[i], axpy_ver[i]);
        }
    }

    free(tempa);
    free(tempb);
    free(vector_add_res);
    free(vector_sub_res);
    free(vector_mul_res);
    free(axpy_res);
    free(add_res);
    free(sub_res);
    free(mul_res);
    free(axpy_r);

    free(twd_s);
    free(x_s);
    free(vvadd_ver);
    free(vvsub_ver);
    free(vvmul_ver);
    free(axpy_ver);
    
    fclose(fp);
    
    char add_avg_str[50];
    char sub_avg_str[50];
    char mul_avg_str[50];
    char axpy_avg_str[50];
    
    sprintf(add_avg_str, "%.3f", add_avg);
    sprintf(sub_avg_str, "%.3f", sub_avg);
    sprintf(mul_avg_str, "%.3f", mul_avg);
    sprintf(axpy_avg_str, "%.3f", axpy_avg);
    
    printf("AVX512 BLAS:\n");
    printf("%-25s %-25s %-25s %-25s\n", "Vector Add [ns]", "Vector Subtraction [ns]", "Vector Multiply [ns]", "AXPY [ns]");
    printf("%-25s %-25s %-25s %-25s\n", add_avg_str, sub_avg_str, mul_avg_str, axpy_avg_str);

}


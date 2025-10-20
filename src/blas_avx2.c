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

__m256i one, negone, t;

__attribute__((always_inline)) void addmod128(__m256i* ch, __m256i* cl, __m256i ah, __m256i al, __m256i bh, __m256i bl, __m256i mh, __m256i ml) {
    __m256i t30, t28, t29, d1, d2, d3; 
    __m256i q1_m, q2_m, c1_m, q3_m, q4_m, c2_m, a31_m, a35_m, a38_m, a34_m, i27_m, a41_m, a44_m, a43_m, i28_m, b1_m;

    t30 = _mm256_add_epi64(al, bl);  
    q1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(al,t), _mm256_add_epi64(t30, t));
    q2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(bl, t), _mm256_add_epi64(t30, t));
    c1_m = _mm256_and_si256(_mm256_or_si256(q1_m, q2_m), one);
    t28 = _mm256_add_epi64 (ah, bh);
    t29 = _mm256_add_epi64 (t28, c1_m);
    q3_m = _mm256_or_si256(_mm256_cmpgt_epi64(_mm256_add_epi64(ah, t), _mm256_add_epi64(t29, t)), _mm256_cmpgt_epi64(_mm256_add_epi64(ah, t), _mm256_add_epi64(t29, t)));
    q4_m = _mm256_or_si256(_mm256_cmpgt_epi64(_mm256_add_epi64(bh, t), _mm256_add_epi64(t29, t)), _mm256_cmpgt_epi64(_mm256_add_epi64(bh, t), _mm256_add_epi64(t29, t)));
    c2_m = _mm256_or_si256(q3_m, q4_m);
    a31_m = _mm256_cmpgt_epi64(_mm256_add_epi64(t29, t), _mm256_add_epi64(mh, t));
    a35_m = _mm256_cmpeq_epi64(t29, mh);
    a38_m = _mm256_or_si256(_mm256_cmpgt_epi64(_mm256_add_epi64(t30, t), _mm256_add_epi64(ml, t)), _mm256_cmpeq_epi64(_mm256_add_epi64(t30, t), _mm256_add_epi64(ml, t)));
    a34_m = _mm256_and_si256(a35_m, a38_m);
    i27_m = _mm256_or_si256(a31_m, a34_m);
    a41_m = c2_m;

    a44_m = _mm256_xor_si256(a41_m, negone);

    a43_m = _mm256_and_si256(a44_m, i27_m);

    i28_m = _mm256_or_si256(a41_m, a43_m);
    __m256i i28_n = _mm256_xor_si256(i28_m, negone);

    d1 = _mm256_sub_epi64(t30, ml);
    b1_m = _mm256_and_si256(_mm256_cmpgt_epi64(_mm256_add_epi64(ml, t), _mm256_add_epi64(t30, t)), one);
    d2 = _mm256_sub_epi64(t29, mh);
    d3 = _mm256_sub_epi64(d2, b1_m);

    __m256i t29_m = _mm256_and_si256(i28_n, t29);
    __m256i d3_m = _mm256_and_si256(i28_m, d3);
    *ch = _mm256_or_si256(t29_m, d3_m);

    __m256i t30_m = _mm256_and_si256(i28_n, t30);
    __m256i d1_m = _mm256_and_si256(i28_m, d1);
    *cl = _mm256_or_si256(t30_m, d1_m);
}

__attribute__((always_inline)) void submod128(__m256i* ch, __m256i* cl, __m256i ah, __m256i al, __m256i bh, __m256i bl, __m256i mh, __m256i ml) {
    __m256i t30, c1, t28, t29, d1, d2, d3, c3;
    __m256i c1_m, i28_m, q1_m, q2_m, c3_m;

    t30 = _mm256_sub_epi64(al, bl);
    c1_m = _mm256_and_si256(_mm256_cmpgt_epi64(_mm256_add_epi64(bl, t), _mm256_add_epi64(al, t)), one);
    t28 = _mm256_add_epi64(bh, c1_m);
    t29 = _mm256_sub_epi64(ah, t28);
    i28_m = _mm256_cmpgt_epi64(_mm256_add_epi64(t28, t), _mm256_add_epi64(ah, t));
    __m256i i28_n = _mm256_xor_si256(i28_m, negone);
    d1 = _mm256_add_epi64(t30, ml);
    q1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(t30, t), _mm256_add_epi64(d1, t));
    q2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(ml, t), _mm256_add_epi64(d1,t));
    c3_m = _mm256_and_si256(_mm256_or_si256(q1_m, q2_m), one);
    d2 = _mm256_add_epi64(t29, c3_m);
    d3 = _mm256_add_epi64(d2, mh);

    __m256i t29_m = _mm256_and_si256(i28_n, t29);
    __m256i d3_m = _mm256_and_si256(i28_m, d3);
    *ch = _mm256_or_si256(t29_m, d3_m);
    __m256i t30_m = _mm256_and_si256(i28_n, t30);
    __m256i d1_m = _mm256_and_si256(i28_m, d1);
    *cl = _mm256_or_si256(t30_m, d1_m);
}

__attribute__((always_inline)) void mul64(__m256i* ch, __m256i* cl, __m256i a, __m256i b) {
    __m256i c1, c2, cx, cc1, ch1, ch2, cc, ch3, cl1, cl1h, cxl, cc2, ch11, cha;
    __m256i cc1a_m, cc1b_m, cc1_m, cc2_m;

    uint64_t temp;

    __m256i ah = _mm256_srli_epi64(a, 32);
    __m256i temp_mask = _mm256_set1_epi64x(0xFFFFFFFF);
    __m256i al = _mm256_and_si256(a, temp_mask);

    __m256i bh = _mm256_srli_epi64(b, 32);
    __m256i bl = _mm256_and_si256(b, temp_mask);

    c1 = _mm256_mul_epu32(ah, bl);

    c2 = _mm256_mul_epu32(al, bh);

    cx = _mm256_add_epi64(c1, c2);
    cc1a_m = _mm256_cmpgt_epi64(_mm256_add_epi64(c1, t), _mm256_add_epi64(cx, t));
    cc1b_m = _mm256_cmpgt_epi64(_mm256_add_epi64(c2, t), _mm256_add_epi64(cx, t));
    cc1_m = _mm256_or_si256(cc1a_m, cc1b_m);
    ch1 = _mm256_mul_epu32(ah, bh);
    ch2 = _mm256_srli_epi64(cx, 32);
    temp_mask = _mm256_slli_epi64(one, 32);
    cc = _mm256_and_si256(cc1_m, temp_mask);
    ch3 = _mm256_or_si256(cc, ch2);

    __m256i s = _mm256_add_epi64(c1, c2);
    __m256i albl = _mm256_mul_epu32(al, bl);
    cl1 = _mm256_add_epi64(albl, _mm256_slli_epi64(s, 32));
    cl1h = _mm256_srli_epi64(cl1, 32);

    temp_mask = _mm256_set1_epi64x(0xFFFFFFFF);
    cxl = _mm256_and_si256(cx, temp_mask);
    cc2_m = _mm256_and_si256(_mm256_cmpgt_epi64(_mm256_add_epi64(cxl, t), _mm256_add_epi64(cl1h, t)), one);
    ch11 = _mm256_add_epi64(ch1, cc2_m);
    cha = _mm256_add_epi64(ch11, ch3);

    *ch = cha;
    *cl = cl1;
}

__attribute__((always_inline)) void mullo64(__m256i* c, __m256i a, __m256i b) {
    uint64_t temp;

    __m256i ah = _mm256_srli_epi64(a, 32);
    __m256i temp_mask = _mm256_set1_epi64x(0xFFFFFFFF);
    __m256i al = _mm256_and_si256(a, temp_mask);

    __m256i bh = _mm256_srli_epi64(b, 32);
    __m256i bl = _mm256_and_si256(b, temp_mask);

    __m256i albl = _mm256_mul_epu32(al, bl);
    __m256i ahbl = _mm256_mul_epu32(ah, bl);
    __m256i albh = _mm256_mul_epu32(al, bh);

    __m256i s = _mm256_add_epi64(ahbl, albh);
    *c = _mm256_add_epi64(albl, _mm256_slli_epi64(s, 32));
}

__attribute__((always_inline)) void mulhi64(__m256i* c, __m256i a, __m256i b) {
    __m256i c1, c2, cx, cc1, ch1, ch2, cc, ch3, cl1, cl1h, cxl, cc2, ch11, cha;
    __m256i cc1a_m, cc1b_m, cc1_m, cc2_m;

    uint64_t temp;

    __m256i ah = _mm256_srli_epi64(a, 32);
    __m256i temp_mask = _mm256_set1_epi64x(0xFFFFFFFF);
    __m256i al = _mm256_and_si256(a, temp_mask);

    __m256i bh = _mm256_srli_epi64(b, 32);
    __m256i bl = _mm256_and_si256(b, temp_mask);

    c1 = _mm256_mul_epu32(ah, bl);

    c2 = _mm256_mul_epu32(al, bh);

    cx = _mm256_add_epi64(c1, c2);
    cc1a_m = _mm256_cmpgt_epi64(_mm256_add_epi64(c1, t), _mm256_add_epi64(cx, t));
    cc1b_m = _mm256_cmpgt_epi64(_mm256_add_epi64(c2, t), _mm256_add_epi64(cx, t));
    cc1_m = _mm256_or_si256(cc1a_m, cc1b_m);
    ch1 = _mm256_mul_epu32(ah, bh);
    ch2 = _mm256_srli_epi64(cx, 32);
    temp_mask = _mm256_slli_epi64(one, 32);
    cc = _mm256_and_si256(cc1_m, temp_mask);
    ch3 = _mm256_or_si256(cc, ch2);

    __m256i s = _mm256_add_epi64(c1, c2);
    __m256i albl = _mm256_mul_epu32(al, bl);
    cl1 = _mm256_add_epi64(albl, _mm256_slli_epi64(s, 32));
    cl1h = _mm256_srli_epi64(cl1, 32);

    temp_mask = _mm256_set1_epi64x(0xFFFFFFFF);
    cxl = _mm256_and_si256(cx, temp_mask);
    cc2_m = _mm256_and_si256(_mm256_cmpgt_epi64(_mm256_add_epi64(cxl, t), _mm256_add_epi64(cl1h, t)), one);
    ch11 = _mm256_add_epi64(ch1, cc2_m);
    cha = _mm256_add_epi64(ch11, ch3);

    *c = cha;
}


#ifdef USE_KARATSUBA
    __attribute__((always_inline)) void mul128(__m256i* chh, __m256i* chl, __m256i* clh, __m256i* cll, __m256i ah, __m256i al, __m256i bh, __m256i bl) {
        __m256i xh1, xl, yh, yl, zh, zl;
        __m256i ul, vl, xh2a, xh2b, xh2, xc1, xc2, xh, xc3, xc1a, xc;
        __m256i uc1_m, uc2_m, uc_m, vc1_m, vc2_m, vc_m, xc11_m, xc12_m, xc1_m, xc31_m, xc32_m, xc3_m;

        __m256i ql, qc1, yh1, qh, qc, rl, rlc, qh1, rh, rc1, rc2, rc, clc, wl, wlc, wh, chlc, wh1;
        __m256i qc11_m, qc12_m, qc1_m, qca_m, qcb_m, qc_m, rlc_m, rc1_m, clc1_m, clc2_m, clc_m, wlc1_m, wlc2_m, wlc_m, chlc1_m, chlc2_m, chlc_m;

        uint64_t temp;

        ul = _mm256_add_epi64(ah, al);
        uc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(al, t), _mm256_add_epi64(ul, t));
        uc2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(ah, t), _mm256_add_epi64(ul, t));
        uc_m = _mm256_or_si256(uc1_m, uc2_m);

        vl = _mm256_add_epi64(bh, bl);
        vc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(bl, t), _mm256_add_epi64(vl,t));
        vc2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(bh, t), _mm256_add_epi64(vl,t));
        vc_m = _mm256_or_si256(vc1_m, vc2_m);
 
        mul64(&xh1, &xl, ul, vl);

        xh2a = _mm256_and_si256(uc_m, vl);
        xh2b = _mm256_and_si256(vc_m, ul);
        xh2 = _mm256_add_epi64(xh2a, xh2b);
        xc11_m = _mm256_cmpgt_epi64(_mm256_add_epi64(xh2a, t), _mm256_add_epi64(xh2,t));
        xc12_m = _mm256_cmpgt_epi64(_mm256_add_epi64(xh2b, t), _mm256_add_epi64(xh2,t));
        xc1_m = _mm256_and_si256(_mm256_or_si256(xc11_m, xc12_m), one);
        __m256i xc2_m = _mm256_and_si256(_mm256_and_si256(uc_m, vc_m), one);
        xh = _mm256_add_epi64(xh1, xh2);
        
        xc31_m = _mm256_cmpgt_epi64(_mm256_add_epi64(xh1, t), _mm256_add_epi64(xh, t)); 
        xc32_m = _mm256_cmpgt_epi64(_mm256_add_epi64(xh2, t), _mm256_add_epi64(xh, t)); 
        xc3_m = _mm256_and_si256(_mm256_or_si256(xc31_m, xc32_m), one);
        xc1a = _mm256_add_epi64(xc1_m, xc3_m);
        xc = _mm256_add_epi64(xc1a, xc2_m);

        mul64(&yh, &yl, ah, bh);

        mul64(&zh, &zl, al, bl);
        
        ql = _mm256_add_epi64(yl, zl);
        qc11_m = _mm256_cmpgt_epi64(_mm256_add_epi64(yl, t), _mm256_add_epi64(ql, t));
        qc12_m = _mm256_cmpgt_epi64(_mm256_add_epi64(zl, t), _mm256_add_epi64(ql, t));
        qc1_m = _mm256_and_si256(_mm256_or_si256(qc11_m, qc12_m), one);
        yh1 = _mm256_add_epi64(yh, qc1_m);
        qh = _mm256_add_epi64(yh1, zh);
        qca_m = _mm256_cmpgt_epi64(_mm256_add_epi64(yh1, t), _mm256_add_epi64(qh, t));
        qcb_m = _mm256_cmpgt_epi64(_mm256_add_epi64(zh, t), _mm256_add_epi64(qh, t));
        qc_m = _mm256_or_si256(qca_m, qcb_m);
        rl = _mm256_sub_epi64(xl, ql);
        rlc_m = _mm256_cmpgt_epi64(_mm256_add_epi64(ql, t), _mm256_add_epi64(xl, t));
        qh1 = _mm256_add_epi64(qh, _mm256_and_si256(rlc_m, one));
        rh = _mm256_sub_epi64(xh, qh1);

        rc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(qh1, t), _mm256_add_epi64(xh, t));
        rc2 = _mm256_add_epi64(xc, qc_m);
        rc = _mm256_add_epi64(rc2, rc1_m);

        *cll = zl;

        *clh = _mm256_add_epi64(zh, rl);

        clc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(zh, t), _mm256_add_epi64(*clh, t));
        clc2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(rl, t), _mm256_add_epi64(*clh, t));
        clc_m = _mm256_and_si256(_mm256_or_si256(clc1_m, clc2_m), one);
        wl = _mm256_add_epi64(rh, clc_m);
        
        wlc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(rh, t), _mm256_add_epi64(wl, t));
        wlc2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(clc, t), _mm256_add_epi64(wl, t));
        wlc_m = _mm256_and_si256(_mm256_or_si256(wlc1_m, wlc2_m), one);
        wh = _mm256_add_epi64(rc, wlc_m);

        *chl = _mm256_add_epi64(yl, wl);

        chlc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(yl, t), _mm256_add_epi64(*chl, t)); 
        chlc2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(wl, t), _mm256_add_epi64(*chl, t));
        chlc_m = _mm256_and_si256(_mm256_or_si256(chlc1_m, chlc2_m), one);
        wh1 = _mm256_add_epi64(wh, chlc_m);

        *chh = _mm256_add_epi64(yh, wh1);
    }

#else
    __attribute__((always_inline)) void mul128(__m256i* chh, __m256i* chl, __m256i* clh, __m256i* cll, __m256i ah, __m256i al, __m256i bh, __m256i bl) {

        __m256i alblh, albll, albhh, albhl, ahblh, ahbll, ahbhh, ahbhl;
        __m256i cxl, cxlc, cxh, cx2l, cx2c, cx2h, chxl, chxc, chxh;
        __m256i cxlc1_m, cxlc2_m, cxlc_m, cx2c1_m, cx2c2_m, cx2c_m, chxc1_m, chxc2_m, chxc_m;

        mul64(&alblh, &albll, al, bl);
        mul64(&albhh, &albhl, al, bh);
        mul64(&ahblh, &ahbll, ah, bl);
        mul64(&ahbhh, &ahbhl, ah, bh);

        cxl = _mm256_add_epi64(albhl, ahbll);
        cxlc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(albhl, t), _mm256_add_epi64(cxl, t)); 
        cxlc2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(ahbll, t), _mm256_add_epi64(cxl, t));
        cxlc_m = _mm256_and_si256(_mm256_or_si256(cxlc1_m, cxlc2_m), one); 
        cxh = _mm256_add_epi64(albhh, ahblh);
        cx2l = _mm256_add_epi64(cxl, alblh);
        cx2c1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(cxl, t), _mm256_add_epi64(cx2l,t));
        cx2c2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(alblh, t), _mm256_add_epi64(cx2l,t));
        cx2c_m = _mm256_and_si256(_mm256_or_si256(cx2c1_m, cx2c2_m), one);
        cx2h = _mm256_add_epi64(cxh, cx2c_m);
        cx2h = _mm256_add_epi64(cx2h, cxlc_m);
        chxl = _mm256_add_epi64(ahbhl, cx2h);
        chxc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(ahbhl,t), _mm256_add_epi64(chxl,t));
        chxc2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(cx2h,t), _mm256_add_epi64(chxl,t));
        chxc_m = _mm256_and_si256(_mm256_or_si256(chxc1_m, chxc2_m), one); 
        chxh = _mm256_add_epi64(ahbhh, chxc_m);

        *chh = chxh;
        *chl = chxl;
        *clh = cx2l;
        *cll = albll;
    }
#endif

__attribute__((always_inline)) void mullo128(__m256i* ch, __m256i* cl, __m256i ah, __m256i al, __m256i bh, __m256i bl) {
    __m256i alblh, albll, albhl, ahbll;
    __m256i cxl, cx2l;

    uint64_t temp;
    mul64(&alblh, &albll, al, bl);

    mullo64(&albhl, al, bh);

    mullo64(&ahbll, ah, bl);
    
    cxl = _mm256_add_epi64(albhl, ahbll);
    cx2l = _mm256_add_epi64(cxl, alblh);

    *ch = cx2l;
    *cl = albll;
}

__attribute__((always_inline)) void mulhi128(__m256i* ch, __m256i* cl, __m256i ah, __m256i al, __m256i bh, __m256i bl) {
    __m256i alblh, albll, albhh, albhl, ahblh, ahbll, ahbhh, ahbhl;
    __m256i cxl, cxlc, cxh, cx2l, cx2c, cx2h, chxl, chxc, chxh;
    __m256i cxlc1_m, cxlc2_m, cxlc_m, cx2c1_m, cx2c2_m, cx2c_m, chxc1_m, chxc2_m, chxc_m;

    uint64_t temp;
    mulhi64(&alblh, al, bl);
    mul64(&albhh, &albhl, al, bh);
    mul64(&ahblh, &ahbll, ah, bl);
    mul64(&ahbhh, &ahbhl, ah, bh);
   
    cxl = _mm256_add_epi64(albhl, ahbll);
    cxlc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(albhl, t), _mm256_add_epi64(cxl, t)); 
    cxlc2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(ahbll, t), _mm256_add_epi64(cxl, t));
    cxlc_m = _mm256_and_si256(_mm256_or_si256(cxlc1_m, cxlc2_m), one); 
    cxh = _mm256_add_epi64(albhh, ahblh);
    cx2l = _mm256_add_epi64(cxl, alblh);
    cx2c1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(cxl, t), _mm256_add_epi64(cx2l,t));
    cx2c2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(alblh, t), _mm256_add_epi64(cx2l,t));
    cx2c_m = _mm256_and_si256(_mm256_or_si256(cx2c1_m, cx2c2_m), one);
    cx2h = _mm256_add_epi64(cxh, cx2c_m);
    cx2h = _mm256_add_epi64(cx2h, cxlc_m);
    chxl = _mm256_add_epi64(ahbhl, cx2h);
    chxc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(ahbhl,t), _mm256_add_epi64(chxl,t));
    chxc2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(cx2h,t), _mm256_add_epi64(chxl,t));
    chxc_m = _mm256_and_si256(_mm256_or_si256(chxc1_m, chxc2_m), one); 
    chxh = _mm256_add_epi64(ahbhh, chxc_m);

    *ch = chxh;
    *cl = chxl;
}

__attribute__((always_inline)) void mulmod128(__m256i* ch, __m256i* cl, __m256i ah, __m256i al, __m256i bh, __m256i bl, __m256i mh, __m256i ml, __m256i muh, __m256i mul) {
    __m256i qll, qlh, qhl, qhh;
    __m256i tmpll, tmplh, qlla, qllb, qll1, qlha, qlhb, qlh1, qhq;

    uint64_t temp;

    mul128(&qhh, &qhl, &qlh, &qll, ah, al, bh, bl);
    
    tmpll = qll;
    tmplh = qlh;
    qlla = _mm256_srli_epi64(qlh, (MBITS-2-64));
    qllb = _mm256_slli_epi64(qhl, (128 - MBITS + 2));
    qll1 = _mm256_or_si256(qlla, qllb);
    qlha = _mm256_srli_epi64(qhl, (MBITS - 2 - 64));
    qlhb = _mm256_slli_epi64(qhh, (128 - MBITS + 2));
    qlh1 = _mm256_or_si256(qlha, qlhb);
    qhq = _mm256_srli_epi64(qhh, (MBITS - 2 - 64));

    __m256i qmh1h, qmh1l, qmh2, qmhl, qmhc, qmhh, qql1, qql2, qql, qqh;
    __m256i qmhc1_m, qmhc2_m, qmhc_m;
    mulhi128(&qmh1h, &qmh1l, qlh1, qll1, muh, mul);

    mullo64(&qmh2, qhq, mul);
    qmhl = _mm256_add_epi64(qmh1l, qmh2);
    qmhc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(qmh1l, t), _mm256_add_epi64(qmhl,t));
    qmhc2_m = _mm256_cmpgt_epi64(_mm256_add_epi64(qmh2, t), _mm256_add_epi64(qmhl,t));
    qmhc_m = _mm256_and_si256(_mm256_or_si256(qmhc1_m, qmhc2_m), one);
    qmhh = _mm256_add_epi64(qmh1h, qmhc_m);
    qql1 = _mm256_srli_epi64(qmhl, 1);
    qql2 = _mm256_slli_epi64(qmhh, 63);
    qql = _mm256_or_si256(qql1, qql2);
    qqh = _mm256_srli_epi64(qmhh, 1);

    __m256i qqmll, qqmlh;
    __m256i tmpll1, tmplc, tmplh1a, tmplh1;
    __m256i tmplc1_m, tmplc2_m, tmplc_m;
    mullo128(&qqmlh, &qqmll, qqh, qql, mh, ml);

    tmpll1 = _mm256_sub_epi64(tmpll, qqmll);
    tmplc_m = _mm256_cmpgt_epi64(_mm256_add_epi64(qqmll, t), _mm256_add_epi64(tmpll,t));
    tmplh1a = _mm256_sub_epi64(tmplh, qqmlh);
    tmplh1 = _mm256_add_epi64(tmplh1a, tmplc_m);

    __m256i cc1_m, cc2_m, cc3_m, cc4_m, cc_m;
    cc1_m = _mm256_cmpgt_epi64(_mm256_add_epi64(mh, t), _mm256_add_epi64(tmplh1, t));
    cc2_m = _mm256_cmpeq_epi64(tmplh1, mh);
    cc3_m = _mm256_cmpgt_epi64(_mm256_add_epi64(ml, t), _mm256_add_epi64(tmplh1, t));
    cc4_m = _mm256_and_si256(cc2_m, cc3_m);
    cc_m = _mm256_or_si256(cc1_m, cc4_m);
    __m256i cc_n = _mm256_xor_si256(cc_m, negone);

    __m256i tmplml, tmplmha, tmplmh;
    __m256i tmplmc1_m, tmplmc2_m;
    tmplml = _mm256_sub_epi64(tmpll1, ml);
    tmplc_m = _mm256_cmpgt_epi64(_mm256_add_epi64(ml, t), _mm256_add_epi64(tmpll1,t));
    tmplmha = _mm256_sub_epi64(tmplh1, mh);
    tmplmh = _mm256_add_epi64(tmplmha, tmplc_m);
    __m256i tmpll1_m = _mm256_and_si256(cc_m, tmpll1);
    __m256i tmplml_m = _mm256_and_si256(cc_n, tmplml);
    *cl = _mm256_or_si256(tmpll1_m, tmplml_m);

    __m256i tmplh1_m = _mm256_and_si256(cc_m, tmplh1);
    __m256i tmplmh_m = _mm256_and_si256(cc_n, tmplmh);
    *ch = _mm256_or_si256(tmplh1_m, tmplmh_m);
}

__attribute__((always_inline)) void vector_add(__m256i* c, __m256i* a, __m256i* b, __m256i mh_256, __m256i ml_256){
    __asm volatile("# LLVM-MCA-BEGIN vector add":::"memory");
    __m256i ch_256, cl_256, ah_256, al_256, bh_256, bl_256;

    for(int i = 0; i < 1024/4; i++){
        ah_256 = _mm256_unpacklo_epi64(a[2*i], a[2*i+1]);
        al_256 = _mm256_unpackhi_epi64(a[2*i], a[2*i+1]);

        bh_256 = _mm256_unpacklo_epi64(b[2*i], b[2*i+1]);
        bl_256 = _mm256_unpackhi_epi64(b[2*i], b[2*i+1]);

        addmod128(&ch_256, &cl_256, ah_256, al_256, bh_256, bl_256, mh_256, ml_256);

        c[2*i] = _mm256_unpacklo_epi64(ch_256, cl_256);
        c[2*i+1] = _mm256_unpackhi_epi64(ch_256, cl_256);
    }
    __asm volatile("# LLVM-MCA-END vector add":::"memory");
}

__attribute__((always_inline)) void vector_sub(__m256i* c, __m256i* a, __m256i* b, __m256i mh_256, __m256i ml_256){
    __asm volatile("# LLVM-MCA-BEGIN vector sub":::"memory");
    __m256i ch_256, cl_256, ah_256, al_256, bh_256, bl_256;

    for(int i = 0; i < 1024/4; i++){
        ah_256 = _mm256_unpacklo_epi64(a[2*i], a[2*i+1]);
        al_256 = _mm256_unpackhi_epi64(a[2*i], a[2*i+1]);

        bh_256 = _mm256_unpacklo_epi64(b[2*i], b[2*i+1]);
        bl_256 = _mm256_unpackhi_epi64(b[2*i], b[2*i+1]); 

        submod128(&ch_256, &cl_256, ah_256, al_256, bh_256, bl_256, mh_256, ml_256);

        c[2*i] = _mm256_unpacklo_epi64(ch_256, cl_256);
        c[2*i+1] = _mm256_unpackhi_epi64(ch_256, cl_256);
    }
    __asm volatile("# LLVM-MCA-END vector sub":::"memory");
}

__attribute__((always_inline)) void vector_mul(__m256i* c, __m256i* a, __m256i* b, __m256i mh_256, __m256i ml_256, __m256i muh_256, __m256i mul_256){
    __asm volatile("# LLVM-MCA-BEGIN vector mul":::"memory");
    __m256i ch_256, cl_256, ah_256, al_256, bh_256, bl_256;

    for(int i = 0; i < 1024/4; i++){
        ah_256 = _mm256_unpacklo_epi64(a[2*i], a[2*i+1]);
        al_256 = _mm256_unpackhi_epi64(a[2*i], a[2*i+1]);

        bh_256 = _mm256_unpacklo_epi64(b[2*i], b[2*i+1]);
        bl_256 = _mm256_unpackhi_epi64(b[2*i], b[2*i+1]);

        mulmod128(&ch_256, &cl_256, ah_256, al_256, bh_256, bl_256, mh_256, ml_256, muh_256, mul_256);

        c[2*i] = _mm256_unpacklo_epi64(ch_256, cl_256);
        c[2*i+1] = _mm256_unpackhi_epi64(ch_256, cl_256);
    }
    __asm volatile("# LLVM-MCA-END vector mul":::"memory");
}

__attribute__((always_inline)) void axpy(__m256i* c, __m256i* a, __m256i* b, __m256i xh_256, __m256i xl_256, __m256i mh_256, __m256i ml_256, __m256i muh_256, __m256i mul_256){
    __asm volatile("# LLVM-MCA-BEGIN axpy":::"memory");
    __m256i temph, templ, ch_256, cl_256, ah_256, al_256, bh_256, bl_256;

    for(int i = 0; i < 1024/4; i++){
        ah_256 = _mm256_unpacklo_epi64(a[2*i], a[2*i+1]);
        al_256 = _mm256_unpackhi_epi64(a[2*i], a[2*i+1]);

        bh_256 = _mm256_unpacklo_epi64(b[2*i], b[2*i+1]);
        bl_256 = _mm256_unpackhi_epi64(b[2*i], b[2*i+1]);

        mulmod128(&temph, &templ, xh_256, xl_256, bh_256, bl_256, mh_256, ml_256, muh_256, mul_256);
        addmod128(&ch_256, &cl_256, ah_256, al_256, temph, templ, mh_256, ml_256);

        c[2*i] = _mm256_unpacklo_epi64(ch_256, cl_256);
        c[2*i+1] = _mm256_unpackhi_epi64(ch_256, cl_256);
    }
    __asm volatile("# LLVM-MCA-END axpy":::"memory");
}

int main() {
    negone = _mm256_set1_epi64x(-1);
    one = _mm256_set1_epi64x(1);
    uint64_t temp = ((1 << 63) - 1)+1;
    t = _mm256_set1_epi64x(-9223372036854775808);

    uint128_t m, mu, scalar;
    m = MODULUS;
    mu = MU;
    scalar = INT128(208488538572646751, 12808410693871070164);

    __m256i mh_256, ml_256, muh_256, mul_256;
    __m256i sh_256, sl_256;  

    uint64_t mh = HI64(m);
    uint64_t ml = LO64(m);

    uint64_t muh = HI64(mu);
    uint64_t mul = LO64(mu);

    uint64_t sh = HI64(scalar);
    uint64_t sl = LO64(scalar);

    mh_256 = _mm256_set1_epi64x(mh);
    ml_256 = _mm256_set1_epi64x(ml);

    muh_256 = _mm256_set1_epi64x(muh);
    mul_256 = _mm256_set1_epi64x(mul);

    sh_256 = _mm256_set1_epi64x(sh);
    sl_256 = _mm256_set1_epi64x(sl);

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

    __m256i* tempa = (__m256i*)(aligned_alloc(64, sizeof(__m256i) * (2048/4)));
    __m256i* tempb = (__m256i*)(aligned_alloc(64, sizeof(__m256i) * (2048/4)));
    for(int j = 0; j < 2048/4; j++){
        tempa[j] = _mm256_set_epi64x(twd_s[3 + j*4], twd_s[2 + j*4], twd_s[1 + j*4], twd_s[j*4]);
        tempb[j] = _mm256_set_epi64x(x_s[3 + j*4], x_s[2 + j*4], x_s[1 + j*4], x_s[j*4]);
    }
    

    __m256i* vector_add_res = (__m256i*)(aligned_alloc(64, sizeof(__m256i) * (2048/4)));
    __m256i* vector_sub_res = (__m256i*)(aligned_alloc(64, sizeof(__m256i) * (2048/4)));
    __m256i* vector_mul_res = (__m256i*)(aligned_alloc(64, sizeof(__m256i) * (2048/4)));
    __m256i* axpy_res = (__m256i*)(aligned_alloc(64, sizeof(__m256i) * (2048/4)));

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

    fprintf(fp, "vector add: \n");
    for(int i = 0; i < 100; i++){
        struct timespec start;
        struct timespec end;
        timespec_get(&start, TIME_UTC);
        
        vector_add(vector_add_res, tempa, tempb, mh_256, ml_256);
        
        timespec_get(&end, TIME_UTC);
        fprintf(fp, "0.%09ld\n", ((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)));
        if(i >= 50){
            add_avg += ((double)((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)) / (1024.0));
        }
    }
    add_avg /= 50.0;

    fprintf(fp, "\n\n\nvector sub: \n");
    for(int i = 0; i < 100; i++){
        struct timespec start;
        struct timespec end;
        timespec_get(&start, TIME_UTC);
        
        vector_sub(vector_sub_res, tempa, tempb, mh_256, ml_256);

        timespec_get(&end, TIME_UTC);
        fprintf(fp, "0.%09ld\n", ((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)));
        if(i >= 50){
            sub_avg += ((double)((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)) / (1024.0));
        }
    }
    sub_avg /= 50.0;

    fprintf(fp, "\n\n\nvector mul: \n");
    for(int i = 0; i < 100; i++){
        struct timespec start;
        struct timespec end;
        timespec_get(&start, TIME_UTC);
        
        vector_mul(vector_mul_res, tempa, tempb, mh_256, ml_256, muh_256, mul_256);

        timespec_get(&end, TIME_UTC);
        fprintf(fp, "0.%09ld\n", (end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec));
        if(i >= 50){
            mul_avg += ((double)((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)) / (1024.0));
        }
    }
    mul_avg /= 50.0;

    fprintf(fp, "\n\n\naxpy: \n");
    for(int i = 0; i < 100; i++){
        struct timespec start;
        struct timespec end;
        timespec_get(&start, TIME_UTC);
        
        axpy(axpy_res, tempa, tempb, sh_256, sl_256, mh_256, ml_256, muh_256, mul_256);
        
        timespec_get(&end, TIME_UTC);
        fprintf(fp, "0.%09ld\n", ((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)));
        if(i >= 50){
            axpy_avg += ((double)((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)) / (1024.0));
        }
    }
    axpy_avg /= 50.0;

    for(int i = 0; i < 2048/4; i++){
        _mm256_storeu_si256(&(add_res[i*4]), vector_add_res[i]);
        _mm256_storeu_si256(&(sub_res[i*4]), vector_sub_res[i]);
        _mm256_storeu_si256(&(mul_res[i*4]), vector_mul_res[i]);
        _mm256_storeu_si256(&(axpy_r[i*4]), axpy_res[i]);
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
    
    printf("AVX2 BLAS:\n");
    printf("%-25s %-25s %-25s %-25s\n", "Vector Add [ns]", "Vector Subtraction [ns]", "Vector Multiply [ns]", "AXPY [ns]");
    printf("%-25s %-25s %-25s %-25s\n", add_avg_str, sub_avg_str, mul_avg_str, axpy_avg_str);

}


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

#define USE_INT128
// #define USE_KARATSUBA
#define USE_MBIT124

typedef __uint128_t uint128_t;

#define stages 18
#define N (1 << stages)

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

   
#ifdef USE_INT128

    uint128_t addmod128(uint128_t a, uint128_t b, uint128_t m) {
        _Bool c3, c4, c5, i14;
        uint64_t d4, d6, t18, t19, al, ah, bl, bh, cl, ch, ml, mh, s17, s17a;
        uint128_t ax, s16, s18, s18a, cx, dx, c;

        al = (uint64_t)(a);
        ah = (uint64_t)(a >> 64);
        bl = (uint64_t)(b);
        bh = (uint64_t)(b >> 64);
        ml = (uint64_t)(m);
        mh = (uint64_t)(m >> 64);

        s16 = ((uint128_t)al) + ((uint128_t)bl);
        t18 = (uint64_t)s16;
        c3 = (s16 >> 64) & 0x1;
        s18 = ((uint128_t)ah) + ((uint128_t)bh) + ((uint128_t)c3);
        t19 = (uint64_t)s18;
        c4 = (s18 >> 64) & 0x1;
        cx = (((uint128_t)t19) << 64) | ((uint128_t)t18);
        ax = ((uint128_t)ml) - ((uint128_t)t18);
        s17a = (uint64_t)ax;
        c5 = (_Bool)(ax >> 64) & 0x1;
        d4 = -s17a;
        d6 = t19 - mh - (!c5);
        dx = (((uint128_t)d6) << 64) | ((uint128_t)d4);
        s18a = ((uint128_t)mh) - ((uint128_t)t19) - ((uint128_t)c5);
        i14 = (_Bool)(s18a >> 64) & 0x1;
        c = i14 ? dx : cx;

        return c;
    }

    uint128_t submod128(uint128_t a, uint128_t b, uint128_t m) {
        _Bool c1, c3, i28, q1, q2;
        uint64_t d1, d2, d3, t28, t29, t30, al, ah, bl, bh, cl, ch, ml, mh;
        uint128_t s16, s17, s18, cx, cx2, c;

        al = (uint64_t)(a);
        ah = (uint64_t)(a >> 64);
        bl = (uint64_t)(b);
        bh = (uint64_t)(b >> 64);
        ml = (uint64_t)(m);
        mh = (uint64_t)(m >> 64);

        s16 = ((uint128_t)al) - ((uint128_t)bl);
        t30 = (uint64_t)s16;
        c1 = (s16 >> 64) & 0x1;
        s17 = ((uint128_t)ah) - ((uint128_t)bh) - c1;
        t29 = (uint64_t)s17;
        i28 = (s17 >> 64) & 0x1;
        cx = (((uint128_t)t29) << 64) | ((uint128_t)t30);
        cx2 = cx + m;
        c = i28 ? cx2 : cx;

        return c;
    }

    void mul128(uint128_t* ch, uint128_t* cl, uint128_t a, uint128_t b) {
        uint64_t al, ah, bl, bh, cll, clh, chl, chh, cxl;
        uint128_t albl, albh, ahbl, ahbh, cx, cx2, chx;

        al = (uint64_t)(a);
        ah = (uint64_t)(a >> 64);
        bl = (uint64_t)(b);
        bh = (uint64_t)(b >> 64);

        albl = ((uint128_t)al) * ((uint128_t)bl);
        albh = ((uint128_t)al) * ((uint128_t)bh);
        ahbl = ((uint128_t)ah) * ((uint128_t)bl);
        ahbh = ((uint128_t)ah) * ((uint128_t)bh);

        cx = albh + ahbl;
        cx2 = cx + (albl >> 64);
        chx = ahbh + (cx2 >> 64);

        cll = (uint64_t)albl;
        clh = (uint64_t)cx2;
        *ch = chx;
        *cl = (((uint128_t)clh) << 64) | ((uint128_t)cll);
    }

    uint128_t mullo128(uint128_t a, uint128_t b) {
        uint64_t al, ah, bl, bh, cll, clh, cx, cx2, albhl, ahbll;
        uint128_t albl, cl;

        al = (uint64_t)(a);
        ah = (uint64_t)(a >> 64);
        bl = (uint64_t)(b);
        bh = (uint64_t)(b >> 64);

        albl = ((uint128_t)al) * ((uint128_t)bl);
        albhl = al * bh;
        ahbll = ah * bl;
        cx = albhl + ahbll;
        cx2 = cx + (albl >> 64);

        cll = (uint64_t)albl;
        clh = (uint64_t)cx2;
        cl = (((uint128_t)clh) << 64) | ((uint128_t)cll);
        return cl;
    }

    uint128_t mulhi128(uint128_t a, uint128_t b) {
        uint64_t al, ah, bl, bh, cll, clh, chl, chh, cxl, cx2h, alblh;
        uint128_t albh, ahbl, ahbh, cx, cx2, chx, ch;

        al = (uint64_t)(a);
        ah = (uint64_t)(a >> 64);
        bl = (uint64_t)(b);
        bh = (uint64_t)(b >> 64);

        alblh = (((uint128_t)al) * ((uint128_t)bl)) >> 64;
        albh = ((uint128_t)al) * ((uint128_t)bh);
        ahbl = ((uint128_t)ah) * ((uint128_t)bl);
        ahbh = ((uint128_t)ah) * ((uint128_t)bh);

        cx = albh + ahbl;
        cx2 = cx + alblh;
        cx2h = cx2 >> 64;
        chx = ahbh + cx2h;

        return chx;
    }

    uint128_t mulmod128(uint128_t a, uint128_t b, uint128_t m, uint128_t mu) {
        uint128_t ql, qh, tmpl, qqml, qq, qmh, c, qmh1, ql1, ql2;
        uint64_t qhq, qmh2;
        _Bool cc;

        mul128(&qh, &ql, a, b);
        tmpl = ql;

        ql1 = ql >> (MBITS - 2);
        ql2 = qh << (128 - (MBITS - 2));
        ql = ql2 | ql1;
        qhq = (uint64_t)(qh >> (MBITS - 2));

        qmh1 = mulhi128(ql, mu);
        qmh2 = qhq * LO64(mu);

        qmh = qmh1 + qmh2;
        qq = qmh >> 1;

        qqml = mullo128(qq, m);

        tmpl = tmpl - qqml;
        cc = (tmpl < m);
        c = cc ? tmpl : tmpl - m;

        return c;
    }

#else // int64 only
    uint128_t addmod128(uint128_t a, uint128_t b, uint128_t m) {
        _Bool a31, a34, a35, a38, a41, a43, a44, b1, c1, c2, i27, i28, q1, q2, q3, q4;
        uint64_t a32, a36, a39, a47, a49, a51, d1, d2, d3, t28, t29, t30, al, ah, bl, bh, cl, ch, ml, mh; 

        al = LO64(a);      
        ah = HI64(a);
        bl = LO64(b);
        bh = HI64(b);
        ml = LO64(m);
        mh = HI64(m);
        t30 = al + bl;  
        q1 = (t30 < al);   
        q2 = (t30 < bl);
        c1 = q1 || q2;  
        t28 = ah + bh;  
        t29 = t28 + c1;
        q3 = (t29 < ah);
        q4 = (t29 < bh);
        c2 = q3 || q4;
        a31 = (mh < t29);
        a35 = (mh == t29);
        a38 = (ml <= t30);
        a34 = (a35 && a38); 
        i27 = (a31 || a34);            
        i28 = c2 || i27;
        d1 = t30 - ml;
        b1 = !a38;
        d2 = t29 - mh;
        d3 = d2 - b1;
        ch = i28 ? d3 : t29; 
        cl = i28 ? d1 : t30;
        return INT128(ch, cl);
    }

    uint128_t submod128(uint128_t a, uint128_t b, uint128_t m) {
        _Bool b1, c1, i28, q1, q2;
        uint64_t d1, d2, d3, t28, t29, t30, al, ah, bl, bh, cl, ch, ml, mh;

        al = LO64(a);
        ah = HI64(a);
        bl = LO64(b);
        bh = HI64(b);
        ml = LO64(m);
        mh = HI64(m);

        t30 = al - bl;
        b1 = (bl > al);
        t28 = bh + b1;
        t29 = ah - t28;
        i28 = (t28 > ah);
        d1 = t30 + ml;
        q1 = (d1 < t30);
        q2 = (d1 < ml);
        c1 = q1 || q2;
        d2 = t29 + c1;
        d3 = d2 + mh;

        ch = i28 ? d3 : t29;
        cl = i28 ? d1 : t30;

        return INT128(ch, cl);
    }

    void mul64(uint64_t* ch, uint64_t* cl, uint64_t a, uint64_t b) {
        uint64_t al, ah, bl, bh, c1, c2, cc, cc1, cx, cxl, ch1, ch2, ch3, cl1, cl1h, cc2, cc1a, cc1b, ch11, cha;

        al = a & 0xFFFFFFFF;
        ah = a >> 32;
        bl = b & 0xFFFFFFFF;
        bh = b >> 32;

        c1 = ah * bl;
        c2 = al * bh;
        cx = c1 + c2;
        cc1a = (cx < c1);
        cc1b = (cx < c2);
        cc1 = cc1a || cc1b;
        ch1 = ah * bh;
        ch2 = cx >> 32;
        cc = cc1 << 32;
        ch3 = cc | ch2;
        cl1 = al * bl;
        cl1h = cl1 >> 32;
        cxl = cx & 0xFFFFFFFF;
        cc2 = (cl1h < cxl);
        ch11 = ch1 + cc2;
        cha = ch11 + ch3;

        *ch = cha;
        *cl = cl1;
    }

    uint64_t mullo64(uint64_t a, uint64_t b) {
        return a * b;
    }

    uint64_t mulhi64(uint64_t a, uint64_t b) {
        uint64_t al, ah, bl, bh, c1, c2, cc, cc1, cx, cxl, ch1, ch2, ch3, cl1, cl1h, cc2, cc1a, cc1b, ch11, cha;

        al = a & 0xFFFFFFFF;
        ah = a >> 32;
        bl = b & 0xFFFFFFFF;
        bh = b >> 32;

        c1 = ah * bl;
        c2 = al * bh;
        cx = c1 + c2;
        cc1a = (cx < c1);
        cc1b = (cx < c2);
        cc1 = cc1a || cc1b;
        ch1 = ah * bh;
        ch2 = cx >> 32;
        cc = cc1 << 32;
        ch3 = cc | ch2;
        cl1 = a * b;
        cl1h = cl1 >> 32;
        cxl = cx & 0xFFFFFFFF;
        cc2 = (cl1h < cxl);
        ch11 = ch1 + cc2;
        cha = ch11 + ch3;

        return cha;
    }

    #ifdef USE_KARATSUBA
        void mul128(uint128_t* ch, uint128_t* cl, uint128_t a, uint128_t b) {
            uint64_t al, ah, bl, bh, cll, clh, chl, chh, uc1, uc2, vc1, vc2, xc11, xc12,
                uc, ul, vc, vl, xl, xh, xc, xc1, xh1, xh2, xh2a, xh2b, xc2, xc3, xh3, xc1a, xc31, xc32,
                yh, yl, zh, zl, qc, qh, ql, qc1, yh1, rl, rlc, qh1, rh, rc, clc, wl, wlc, wh, chlc, wh1,
                qc11, qc12, qca, qcb, clc1, clc2, wlc1, wlc2, chlc1, chlc2, rc1, rc2;

            al = LO64(a);
            ah = HI64(a);
            bl = LO64(b);
            bh = HI64(b);

            ul = ah + al;
            uc1 = (ul < al);
            uc2 = (ul < ah);
            uc = uc1 || uc2;
            vl = bh + bl;
            vc1 = (vl < bl);
            vc2 = (vl < bh);
            vc = vc1 || vc2;

            mul64(&xh1, &xl, ul, vl);
            xh2a = ((-(int64_t)uc) & vl);
            xh2b = ((-(int64_t)vc) & ul);
            xh2 = xh2a + xh2b;
            xc11 = (xh2 < xh2a);
            xc12 = (xh2 < xh2b);
            xc1 = xc11 || xc12;
            xc2 = uc & vc;
            xh = xh1 + xh2;
            xc31 = (xh < xh1);
            xc32 = (xh < xh2);
            xc3 = xc31 || xc32;
            xc1a = xc1 + xc3;
            xc = xc1a + xc2;

            mul64(&yh, &yl, ah, bh);
            mul64(&zh, &zl, al, bl);
            
            ql = yl + zl;
            qc11 = (ql < yl);
            qc12 = (ql < zl);
            qc1 = qc11 || qc12;
            yh1 = yh + qc1;
            qh = yh1 + zh;
            qca = (qh < yh1);
            qcb = (qh < zh);
            qc = qca || qcb;
            rl = xl - ql;
            rlc = (ql > xl);
            qh1 = qh + rlc;
            rh = xh - qh1;
            rc1 = (qh1 > xh);
            rc2 = xc - qc;
            rc = rc2 - rc1;
            cll = zl;
            clh = zh + rl;
            clc1 = (clh < zh);
            clc2 = (clh < rl); 
            clc = clc1 || clc2;
            wl = rh + clc;
            wlc1 = (wl < rh);
            wlc2 = (wl < clc);
            wlc = wlc1 || wlc2;
            wh = rc + wlc;
            chl = yl + wl;
            chlc1 = (chl < yl); 
            chlc2 = (chl < wl);
            chlc = chlc1 || chlc2;
            wh1 = wh + chlc;
            chh = yh + wh1;


            *ch = INT128(chh, chl);
            *cl = INT128(clh, cll);
        }
    #else
        void mul128(uint128_t* ch, uint128_t* cl, uint128_t a, uint128_t b) {
            uint64_t al, ah, bl, bh, chl, chh,
                albll, alblh, albhh, albhl, ahblh, ahbll, ahbhh, ahbhl,
                cxl, cxh, cx2l, cx2h, chxl, chxh;
            _Bool cx2c, cx2c1, cx2c2, chxc, chxc1, chxc2, cxlc1, cxlc2, cxlc;

            al = LO64(a);
            ah = HI64(a);
            bl = LO64(b);
            bh = HI64(b);

            mul64(&alblh, &albll, al, bl);
            mul64(&albhh, &albhl, al, bh);
            mul64(&ahblh, &ahbll, ah, bl);
            mul64(&ahbhh, &ahbhl, ah, bh);

            cxl = albhl + ahbll;
            cxlc1 = (cxl < albhl); 
            cxlc2 = (cxl < ahbll);
            cxlc = cxlc1 || cxlc2;
            cxh = albhh + ahblh;
            cx2l = cxl + alblh;
            cx2c1 = (cx2l < cxl);
            cx2c2 = (cx2l < alblh);
            cx2c = cx2c1 || cx2c2;
            cx2h = cxh + cx2c;
            cx2h = cx2h + cxlc;  
            chxl = ahbhl + cx2h;
            chxc1 = (chxl < ahbhl);
            chxc2 = (chxl < cx2h);
            chxc = chxc1 || chxc2;
            chxh = ahbhh + chxc;

            *ch = INT128(chxh, chxl);
            *cl = INT128(cx2l, albll);
        }
    #endif

    uint128_t mullo128(uint128_t a, uint128_t b) {
        uint64_t al, ah, bl, bh, 
            albll, alblh, albhh, albhl, ahblh, ahbll, ahbhl, cxl, cx2l;

        al = LO64(a);
        ah = HI64(a);
        bl = LO64(b);
        bh = HI64(b);

        mul64(&alblh, &albll, al, bl);

        albhl = mullo64(al, bh);
        ahbll = mullo64(ah, bl);

        cxl = albhl + ahbll;
        cx2l = cxl + alblh;

        return INT128(cx2l, albll);
    }

    uint128_t mulhi128(uint128_t a, uint128_t b) {
        uint64_t al, ah, bl, bh, chl, chh,
            albll, alblh, albhh, albhl, ahblh, ahbll, ahbhh, ahbhl,
            cxl, cxh, cx2l, cx2h, chxl, chxh;
        _Bool cx2c, cx2c1, cx2c2, chxc, chxc1, chxc2, cxlc1, cxlc2, cxlc;

        al = LO64(a);
        ah = HI64(a);
        bl = LO64(b);
        bh = HI64(b);

        alblh = mulhi64(al, bl);
        mul64(&albhh, &albhl, al, bh);
        mul64(&ahblh, &ahbll, ah, bl);
        mul64(&ahbhh, &ahbhl, ah, bh);

        cxl = albhl + ahbll;
        cxlc1 = (cxl < albhl);  
        cxlc2 = (cxl < ahbll);
        cxlc = cxlc1 || cxlc2;
        cxh = albhh + ahblh;
        cx2l = cxl + alblh;
        cx2c1 = (cx2l < cxl);
        cx2c2 = (cx2l < alblh);
        cx2c = cx2c1 || cx2c2;
        cx2h = cxh + cx2c;
        cx2h = cx2h + cxlc;  
        chxl = ahbhl + cx2h;
        chxc1 = (chxl < ahbhl);
        chxc2 = (chxl < cx2h);
        chxc = chxc1 || chxc2;
        chxh = ahbhh + chxc;

        return INT128(chxh, chxl);
    }

    uint128_t mulmod128(uint128_t a, uint128_t b, uint128_t m, uint128_t mu) {
        uint128_t ql, qh, qqml, qq, qmh1;
        uint64_t qhq, qll, qlh, qhl, qhh, tmpll, tmplh, qlla, qllb, qll1, qlha, qlhb, qlh1, qmh1h, qmh1l, qmhh, qmhl, 
            qmh2, qql, qqh, qql1, qql2, tmpll1, tmplh1, tmplh1a, qqmll, qqmlh, ml, mh, tmplmh, tmplml, tmplmha, cl, ch;
        _Bool cc, qmhc1, qmhc2,  qmhc, tmplc1, tmplc2, tmplc, cc1, cc2, cc3, cc4, tmplmc1, tmplmc2, tmplmc;

        mul128(&qh, &ql, a, b);
        ml = LO64(m);
        mh = HI64(m);
        qll = LO64(ql);
        qlh = HI64(ql);
        qhl = LO64(qh);
        qhh = HI64(qh);

        tmpll = qll;
        tmplh = qlh;
        qlla = qlh >> (MBITS - 2 - 64);
        qllb = qhl << (128 - MBITS + 2);
        qll1 = qlla | qllb;

        qlha = qhl >> (MBITS - 2 - 64);
        qlhb = qhh << (128 - MBITS + 2);
        qlh1 = qlha | qlhb;
        qhq = qhh >> (MBITS - 2 - 64);
        ql = INT128(qlh1, qll1);
        qmh1 = mulhi128(ql, mu);
        qmh2 = mullo64(qhq, LO64(mu));  
        qmh1h = HI64(qmh1);
        qmh1l = LO64(qmh1);

        qmhl = qmh1l + qmh2;
        qmhc1 = (qmhl < qmh1l);
        qmhc2 = (qmhl < qmh2);
        qmhc = qmhc1 || qmhc2;

        qmhh = qmh1h + qmhc;
        qql1 = qmhl >> 1;
        qql2 = qmhh << 63;
        qql = qql1 | qql2;

        qqh = qmhh >> 1;

        qq = INT128(qqh, qql);

        qqml = mullo128(qq, m);
        qqmll = LO64(qqml);
        qqmlh = HI64(qqml);
        tmpll1 = tmpll - qqmll;
        tmplc = (qqmll > tmpll);  
        tmplh1a = tmplh - qqmlh;
        tmplh1 = tmplh1a - tmplc;

        cc1 = (tmplh1 < mh);
        cc2 = (tmplh1 == mh);
        cc3 = (tmpll1 < ml);
        cc4 = cc2 && cc3;
        cc = cc1 || cc4;

        tmplml = tmpll1 - ml; 
        tmplc = (ml > tmpll1);   
        tmplmha = tmplh1 - mh;
        tmplmh = tmplmha - tmplc;

        cl = cc ? tmpll1 : tmplml;
        ch = cc ? tmplh1 : tmplmh;

        return INT128(ch, cl);
    }

#endif

void printconf(void) {
#ifdef USE_KARATSUBA
    printf(", USE_KARATSUBA");
#endif
#ifdef USE_MBIT124
    printf(", MBIT = 124");
#endif
#ifdef USE_INT128
    printf(", USE_INT128\n");
#else
    printf("\n");
#endif
}

__attribute__((always_inline)) void ntt(uint128_t  *Y, uint128_t  *X, uint128_t modulus, uint128_t mu, uint128_t  *twiddles, uint128_t* as0, uint128_t* as1) {

    for(int i = 0; i < N/2; i++){
        uint128_t p;
        p = mulmod128(twiddles[1], X[i+N/2], modulus, mu);
        as0[2*i] = addmod128(X[i], p, modulus);
        as0[2*i+1] = submod128(X[i], p, modulus);
    }

    for(int i = 1; i < stages - 1; i++){
        int t = 1 << i;
        if(i % 2 == 0){
            for(int j = 0; j < N/2; j++){
                uint128_t p;
                p = mulmod128(twiddles[t+(j%t)], as1[j+N/2], modulus, mu);
                as0[2*j] = addmod128(as1[j], p, modulus);
                as0[2*j+1] = submod128(as1[j], p, modulus);
            }
        }
        else{
            for(int j = 0; j < N/2; j++){
                uint128_t p;
                p = mulmod128(twiddles[t+(j%t)], as0[j+N/2], modulus, mu);
                as1[2*j] = addmod128(as0[j], p, modulus);
                as1[2*j+1] = submod128(as0[j], p, modulus);
            }
        }
    }

    if(stages % 2 == 0){
        for(int i = 0; i < N/2; i++){
            uint128_t p;
            p = mulmod128(twiddles[N/2+i], as0[i+N/2], modulus, mu);
            Y[2*i] = addmod128(as0[i], p, modulus);
            Y[2*i+1] = submod128(as0[i], p, modulus);
        }
    }
    else {
        for(int i = 0; i < N/2; i++){
            uint128_t p;
            p = mulmod128(twiddles[N/2+i], as1[i+N/2], modulus, mu);
            Y[2*i] = addmod128(as1[i], p, modulus);
            Y[2*i+1] = submod128(as1[i], p, modulus);
        }
    }
}


int main() {

    uint128_t mu = INT128(9223372036854775808, 5752);
    uint128_t modulus = INT128(1152921504606846975, 18446744073709550897);

    uint128_t* as0 = (uint128_t*)(malloc(sizeof(uint128_t)*(N)));
    uint128_t* as1 = (uint128_t*)(malloc(sizeof(uint128_t)*(N)));

    uint128_t* y = (uint128_t*)(malloc(sizeof(uint128_t)*N));
    
    double avg_runtime;
    
    avg_runtime = 0L;

    FILE *fp;

    fp = fopen("../bin/scalar_ntt.txt", "w");

    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    uint64_t* twd_s = (uint64_t*)(malloc(sizeof(uint64_t)*(N*2)));
    uint64_t* x_s = (uint64_t*)(malloc(sizeof(uint64_t)*(N*2)));
    uint64_t* ver_s = (uint64_t*)(malloc(sizeof(uint64_t)*(N*2)));

    uint128_t* twd = (uint128_t*)(malloc(sizeof(uint128_t)*N));
    uint128_t* x = (uint128_t*)(malloc(sizeof(uint128_t)*N));
    uint128_t* ver = (uint128_t*)(malloc(sizeof(uint128_t)*N));

    load_twiddles(twd_s, N);
    load_test_inputs(x_s, N);

    for(int i = 0; i < N; i ++){
        twd[i] = INT128(twd_s[2*i], twd_s[2*i+1]);
        x[i] = INT128(x_s[2*i], x_s[2*i+1]);
    }

    fprintf(fp, "ntt: %d\n", N);
    for(int j = 0; j < 100; j++){
        struct timespec start;
        struct timespec end;
        timespec_get(&start, TIME_UTC);
        
        ntt(y, x, modulus, mu, twd, as0, as1);
        
        timespec_get(&end, TIME_UTC);
        fprintf(fp, "0.%09ld\n", ((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)));
        if(j >= 50){
            avg_runtime += ((double)((end.tv_nsec + 1000000000 * end.tv_sec) - (start.tv_nsec + 1000000000 * start.tv_sec)) / (N * log2(N) / 2));
        }
    }
    avg_runtime /= 50;

    if (N <= 1024){
        load_test_outputs(ver_s, N);
        for(int i = 0; i < N; i ++){
            ver[i] = INT128(ver_s[2*i], ver_s[2*i+1]);
        }
        int fail = 0;
        for(int i = 0; i < N; i++){
            if(y[i] != ver[i]){
                fprintf(fp, "i: %d\n", i);
                fprintf(fp, "y: 0x%llX%llX\n", HI64(y[i]), LO64(y[i]));
                fprintf(fp, "ver: 0x%llX%llX\n", HI64(ver[i]), LO64(ver[i]));
                fprintf(fp, "fail\n");
                fail++;
            }
        }
        fprintf(fp, "fail: %d\n", fail);
    }
    
    free(as0);
    free(as1);
    free(y);
    free(twd_s);
    free(x_s);
    free(ver_s);
    free(twd);
    free(x);
    free(ver);
    
    fclose(fp);
    
    char avg_runtime_str[50];
    
    sprintf(avg_runtime_str, "%.2f", avg_runtime);
    
    printf("%-25d %-25s\n", N, avg_runtime_str);
}
/* Diagnostic oracle: official Intel C model vs x86 RCP14 and an integer model.
 * No Rust FFI, no production arithmetic selection. Compile without fast-math.
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reference/RECIP14.c"
static uint32_t integer14(uint32_t bits) {
    uint32_t sign=bits&0x80000000u, exp=(bits>>23)&255u, mant=bits&0x7fffffu;
    if (!mant) return sign|((254u-exp)<<23);
    uint32_t i=mant>>17, a=RCP14_Coeff[2*i+1], b=RCP14_Coeff[2*i];
    int32_t d=(int32_t)((mant>>7)&1023u)-512;
    uint32_t s=(uint32_t)((int32_t)(a<<8)-(int32_t)b*d);
    return sign|((253u-exp)<<23)|((s>>2)&0x7fff80u);
}
__attribute__((target("avx512f,avx512vl")))
static void check16(const uint32_t input[16], uint64_t *count, double *worst, uint32_t *witness) {
    float x[16], out[16];
    for (int i=0;i<16;i++) {type32 v;v.u=input[i];x[i]=v.f;}
    _mm512_storeu_ps(out,_mm512_rcp14_ps(_mm512_loadu_ps(x)));
    for (int i=0;i<16;i++) {
        type32 arg,ref,actual;arg.u=input[i];actual.f=out[i];RCP14S(0,&ref,arg);
        uint32_t integer=integer14(input[i]);
        if (actual.u!=ref.u || integer!=ref.u) {
            fprintf(stderr,"FAIL input=%08x hardware=%08x Intel-C=%08x integer=%08x\n",input[i],actual.u,ref.u,integer);
            exit(1);
        }
        double exact=1.0/(double)arg.f;
        double err=fabs(((double)actual.f-exact)/exact);
        if (err>*worst) {*worst=err;*witness=input[i];}
        if (!(err<0x1p-14)) {fprintf(stderr,"FAIL relative bound\n");exit(1);}
        ++*count;
    }
}
static uint32_t parallel_ref(uint32_t bits) {
    type32 x,y,r;x.u=bits;y.u=0x7ef127eau-bits;
    float e=fmaf(-x.f,y.f,1.0f);
    float a=fmaf(y.f,e,y.f), b=fmaf(e,e,1.0f);
    r.f=a*b;return r.u;
}
static int emit(const char *path) {
    FILE *file=fopen(path,"wb");if(!file){perror(path);return 1;}
    unsigned char block[8192];
    for(uint32_t start=0x3f800000u;start<0x40000000u;start+=1024) {
        for(uint32_t i=0;i<1024;i++) {
            type32 arg,ref;arg.u=start+i;RCP14S(0,&ref,arg);
            uint32_t words[2]={parallel_ref(arg.u),ref.u};
            for(int w=0;w<2;w++)for(int b=0;b<4;b++)block[i*8+w*4+b]=(unsigned char)(words[w]>>(b*8));
        }
        if(fwrite(block,1,sizeof(block),file)!=sizeof(block)){perror("write");fclose(file);return 1;}
    }
    if(fclose(file)){perror("close");return 1;}
    printf("Wrote 8388608 normalized (parallel, Intel RCP14) little-endian records to %s; constant 8192-byte buffer\n",path);
    return 0;
}
int main(int argc,char **argv) {
    if(argc==3 && !strcmp(argv[1],"--emit"))return emit(argv[2]);
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512vl")) {
        fputs("RCP14 hardware reference check requires AVX512F+VL\n",stderr);return 2;
    }
    _mm_setcsr((_mm_getcsr()&~0xe040u)|0x1f80u);
    uint64_t count=0;double worst=0;uint32_t witness=0,input[16];
    for (uint32_t start=0x3f800000u;start<0x40000000u;start+=8) {
        for (int i=0;i<8;i++) {input[2*i]=start+(uint32_t)i;input[2*i+1]=input[2*i]|0x80000000u;}
        check16(input,&count,&worst,&witness);
    }
    const uint32_t mant[]={0,1,127,128,0x1ffff,0x20000,0x3fffff,0x400000,0x7ffffe,0x7fffff};
    for (uint32_t e=2;e<=252;e++) for (unsigned m=0;m<sizeof(mant)/sizeof(*mant);m++) {
        if (e==252 && mant[m]) continue;
        for (int i=0;i<16;i++) input[i]=(e<<23)|mant[m]|((i&1)?0x80000000u:0);
        check16(input,&count,&worst,&witness);
    }
    printf("PASS signed normalized mantissas + edge exponents: %llu cases; RCP14 hardware == official Intel C == integer model\n",(unsigned long long)count);
    printf("max relative error %.12e witness %08x (required <2^-14)\n",worst,witness);
    return 0;
}

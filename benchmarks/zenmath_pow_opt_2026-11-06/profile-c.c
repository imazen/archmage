/* Standalone test-only C ABI/control driver. No shipped dependency or Rust FFI.
   Input is the exact POW32LE1 corpus emitted by the Rust study, including f32
   exponent bits. perf profiles/counters are compared on matching inputs; these
   are separate programs, not interleaved C/Rust wall-time confidence intervals. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <fenv.h>
static uint32_t read32(FILE *f) {uint8_t b[4];if(fread(b,1,4,f)!=4)abort();return (uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;}
int main(int argc,char **argv) {
    if(argc!=5)return 2;
    FILE *f=fopen(argv[3],"rb");if(!f)return 3;
    char magic[8];if(fread(magic,1,8,f)!=8||memcmp(magic,"POW32LE1",8))return 4;
    uint64_t n=read32(f);n|=(uint64_t)read32(f)<<32;if(n!=65536)return 5;
    float *x=malloc(n*4),*y=malloc(n*4),*out=malloc(n*4);if(!x||!y||!out)return 6;
    for(uint64_t i=0;i<n;i++){uint32_t a=read32(f),b=read32(f);memcpy(&x[i],&a,4);memcpy(&y[i],&b,4);}fclose(f);
    void *lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){fprintf(stderr,"%s\n",dlerror());return 7;}
    float (*volatile fn)(float,float)=dlsym(lib,argv[2]);if(!fn)return 8;
    if(fesetround(FE_TONEAREST))return 9;
    unsigned passes=(unsigned)strtoul(argv[4],0,10);if(!passes)return 10;
    for(unsigned p=0;p<passes;p++)for(uint64_t i=0;i<n;i++)out[i]=fn(x[i],y[i]);
    uint64_t hash=0;for(uint64_t i=0;i<n;i++){uint32_t b;memcpy(&b,&out[i],4);hash=(hash*UINT64_C(0x100000001b3))^b;}
    /* This checksum only detects accidental dead-code elimination. Exhaustive
       contract hashing uses Zenhash in the independent digest harness. */
    printf("CHECKSUM %llu\n",(unsigned long long)hash);free(x);free(y);free(out);return 0;
}

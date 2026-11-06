// Standalone research driver: MPFR-directed bounds and interleaved scalar costs.
// This is not a shipping dependency or a claim of exhaustive correctness.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fenv.h>
#include <math.h>
#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 4096
#define REPEAT 128
#define ROUNDS 30
static const char *names[] = {"exp2f", "log2f", "expf", "logf", "cbrtf", "sinf", "tanhf", "powf"};
typedef float (*unary)(float);
typedef float (*binary)(float, float);
static uint32_t state = 0x12345678;
static uint32_t random_bits(void) { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static float from_bits(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static int equal(float x, float y) { return bits(x) == bits(y) || (isnan(x) && isnan(y)); }
static void oracle(int op, mpfr_t z, mpfr_t x, mpfr_t y, mpfr_rnd_t rnd) {
 switch(op) {
 case 0: mpfr_exp2(z,x,rnd); break; case 1: mpfr_log2(z,x,rnd); break;
 case 2: mpfr_exp(z,x,rnd); break; case 3: mpfr_log(z,x,rnd); break;
 case 4: mpfr_cbrt(z,x,rnd); break; case 5: mpfr_sin(z,x,rnd); break;
 case 6: mpfr_tanh(z,x,rnd); break; case 7: mpfr_pow(z,x,y,rnd); break;
 }
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t); return t.tv_sec*1e9+t.tv_nsec; }
int main(int argc, char **argv) {
 if(argc < 2) return 2;
 int count=argc-1, failures=0;
 void *handles[count]; void *fn[count][8];
 for(int v=0;v<count;v++) {
  handles[v]=dlopen(argv[v+1],RTLD_NOW|RTLD_LOCAL);
  if(!handles[v]) { fprintf(stderr,"%s\n",dlerror()); return 2; }
  for(int op=0;op<8;op++) { char sym[40]; snprintf(sym,40,"%s%s",strstr(argv[v+1],"libm.so")?"":"cr_",names[op]);
   fn[v][op]=dlsym(handles[v],sym); if(!fn[v][op]) { fprintf(stderr,"missing %s\n",sym); return 2; }
  }
 }
 fesetround(FE_TONEAREST);
 mpfr_t x,y,lo,hi; mpfr_inits2(128,x,y,lo,hi,(mpfr_ptr)0);
 puts("kind,op,variant,round,value");
 for(int op=0;op<8;op++) {
  unsigned mismatches[count]; memset(mismatches,0,sizeof mismatches);
  for(int i=0;i<100000;i++) {
   float a=from_bits(random_bits()),b=from_bits(random_bits());
   if(i<12) { const uint32_t edge[]={0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0x3f800000}; a=from_bits(edge[i]); b=2.4f; }
   mpfr_set_flt(x,a,MPFR_RNDN); mpfr_set_flt(y,b,MPFR_RNDN);
   mpfr_prec_t precision=128;
   float lower,upper;
   do { mpfr_set_prec(lo,precision); mpfr_set_prec(hi,precision);
    oracle(op,lo,x,y,MPFR_RNDD); oracle(op,hi,x,y,MPFR_RNDU);
    lower=mpfr_get_flt(lo,MPFR_RNDN); upper=mpfr_get_flt(hi,MPFR_RNDN); precision*=2;
    if(precision>16384 && !equal(lower,upper)) { fprintf(stderr,"unresolved oracle %s %08x %08x\n",names[op],bits(a),bits(b)); return 3; }
   } while(!equal(lower,upper));
   for(int v=0;v<count;v++) {
    float result=op==7?((binary)fn[v][op])(a,b):((unary)fn[v][op])(a);
    if(!equal(result,lower)) { if(mismatches[v]++<3) fprintf(stderr,"mismatch %s %s %08x %08x got=%08x expected=%08x\n",names[op],argv[v+1],bits(a),bits(b),bits(result),bits(lower)); }
   }
  }
  for(int v=0;v<count;v++) { printf("mismatch,%s,%s,0,%u\n",names[op],argv[v+1],mismatches[v]); if(v<count-1) failures+=mismatches[v]; }
 }
 mpfr_clears(x,y,lo,hi,(mpfr_ptr)0);
 if(failures) return 1;
 float input[N],output[N]; volatile float sink=0;
 for(int op=0;op<8;op++) {
  for(int i=0;i<N;i++) { float t=(random_bits()>>8)*0x1p-24f; input[i]=(op==1||op==3||op==7)?0.001f+16*t:-8+16*t; }
  for(int round=-2;round<ROUNDS;round++) for(int k=0;k<count;k++) {
   int v=(k+round+2)%count; double start=now();
   for(int j=0;j<REPEAT;j++) { for(int i=0;i<N;i++) output[i]=op==7?((binary)fn[v][op])(input[i],2.4f):((unary)fn[v][op])(input[i]); sink=output[j%N]; }
   double elapsed=now()-start;
   if(round>=0) printf("ns_per_element,%s,%s,%d,%.9f\n",names[op],argv[v+1],round,elapsed/(N*REPEAT));
  }
 }
 (void)sink;
 return 0;
}

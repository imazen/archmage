// Standalone MPFR oracle. No C code is linked into zenmath.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpfr.h>
static uint32_t state=0x12345678;
static uint32_t random_bits(void) {state^=state<<13;state^=state>>17;state^=state<<5;return state;}
static double from_bits(uint64_t b) {double x;memcpy(&x,&b,8);return x;}
static uint64_t bits(double x) {uint64_t b;memcpy(&b,&x,8);return b;}
static int equal(double a,double b) {return bits(a)==bits(b)||(isnan(a)&&isnan(b));}
static void oracle(const char *op,mpfr_t z,mpfr_t x,mpfr_t y,mpfr_rnd_t rnd) {
#define OP(name) if(!strcmp(op,#name)) {mpfr_##name(z,x,rnd);return;}
 OP(exp2) OP(exp) OP(log2) OP(log) OP(log10) OP(cbrt) OP(sin) OP(cos) OP(tan) OP(atan) OP(tanh) OP(log1p) OP(expm1)
 if(!strcmp(op,"pow")) {mpfr_pow(z,x,y,rnd);return;}
 if(!strcmp(op,"atan2")) {mpfr_atan2(z,x,y,rnd);return;}
 fprintf(stderr,"unknown function %s\n",op);exit(2);
}
int main(int argc,char **argv) {
 if(argc!=3) {fprintf(stderr,"usage: oracle FUNCTION OUTPUT\n");return 2;}
 const char *op=argv[1];FILE *out=fopen(argv[2],"wb");if(!out)return 2;
 mpfr_t x,y,lo,hi;mpfr_inits2(128,x,y,lo,hi,(mpfr_ptr)0);
 for(unsigned i=0;i<200000;i++) {
  double a=from_bits(((uint64_t)random_bits()<<32)|random_bits()),b=from_bits(((uint64_t)random_bits()<<32)|random_bits());
  if(i>=100000) {double t=(random_bits()>>8)*0x1p-24f;a=(!strncmp(op,"log",3)||!strcmp(op,"pow"))?0.001f+16*t:-8+16*t;b=-8+16*((random_bits()>>8)*0x1p-24f);}
  if(i<144) {const uint64_t edges[]={0,0x8000000000000000ULL,1,0x8000000000000001ULL,0xfffffffffffffULL,0x10000000000000ULL,0x7fefffffffffffffULL,0xffefffffffffffffULL,0x7ff0000000000000ULL,0xfff0000000000000ULL,0x7ff8000000000000ULL,0x3ff0000000000000ULL};a=from_bits(edges[i%12]);b=from_bits(edges[i/12]);}
  mpfr_set_d(x,a,MPFR_RNDN);mpfr_set_d(y,b,MPFR_RNDN);
  mpfr_prec_t precision=128;double lower,upper;
  do {mpfr_set_prec(lo,precision);mpfr_set_prec(hi,precision);oracle(op,lo,x,y,MPFR_RNDD);oracle(op,hi,x,y,MPFR_RNDU);lower=mpfr_get_d(lo,MPFR_RNDN);upper=mpfr_get_d(hi,MPFR_RNDN);precision*=2;
   if(precision>16384&&!equal(lower,upper)) {fprintf(stderr,"unresolved %s %016llx %016llx\n",op,(unsigned long long)bits(a),(unsigned long long)bits(b));return 3;}
  } while(!equal(lower,upper));
  uint64_t words[]={bits(a),bits(b),bits(lower)};
  for(int w=0;w<3;w++) for(int j=0;j<8;j++) if(fputc((words[w]>>(8*j))&255,out)==EOF)return 2;
 }
 mpfr_clears(x,y,lo,hi,(mpfr_ptr)0);if(fclose(out))return 2;return 0;
}

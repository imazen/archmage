// Standalone MPFR oracle. No C code is linked into zenmath.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpfr.h>
static uint32_t state=0x12345678;
static uint32_t random_bits(void) {state^=state<<13;state^=state>>17;state^=state<<5;return state;}
static float from_bits(uint32_t b) {float x;memcpy(&x,&b,4);return x;}
static uint32_t bits(float x) {uint32_t b;memcpy(&b,&x,4);return b;}
static int equal(float a,float b) {return bits(a)==bits(b)||(isnan(a)&&isnan(b));}
static void oracle(const char *op,mpfr_t z,mpfr_t x,mpfr_t y,mpfr_rnd_t rnd) {
#define OP(name) if(!strcmp(op,#name "f")) {mpfr_##name(z,x,rnd);return;}
 OP(exp2) OP(exp) OP(log2) OP(log) OP(log10) OP(cbrt) OP(sin) OP(cos) OP(tan) OP(atan) OP(tanh) OP(log1p) OP(expm1)
 if(!strcmp(op,"powf")) {mpfr_pow(z,x,y,rnd);return;}
 if(!strcmp(op,"atan2f")) {mpfr_atan2(z,x,y,rnd);return;}
 fprintf(stderr,"unknown function %s\n",op);exit(2);
}
int main(int argc,char **argv) {
 if(argc!=3) {fprintf(stderr,"usage: oracle FUNCTION OUTPUT\n");return 2;}
 const char *op=argv[1];FILE *out=fopen(argv[2],"wb");if(!out)return 2;
 mpfr_t x,y,lo,hi;mpfr_inits2(128,x,y,lo,hi,(mpfr_ptr)0);
 for(unsigned i=0;i<200000;i++) {
  float a=from_bits(random_bits()),b=from_bits(random_bits());
  if(i>=100000) {float t=(random_bits()>>8)*0x1p-24f;a=(!strncmp(op,"log",3)||!strcmp(op,"powf"))?0.001f+16*t:-8+16*t;b=-8+16*((random_bits()>>8)*0x1p-24f);}
  if(i<144) {const uint32_t edges[]={0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0x3f800000};a=from_bits(edges[i%12]);b=from_bits(edges[i/12]);}
  if(!strcmp(op,"exp2f") && i>=144 && i<147) {const uint32_t hard[]={0x3b429d37,0xbcf3a937,0xb8d3d026};a=from_bits(hard[i-144]);}
  mpfr_set_flt(x,a,MPFR_RNDN);mpfr_set_flt(y,b,MPFR_RNDN);
  mpfr_prec_t precision=128;float lower,upper;
  do {mpfr_set_prec(lo,precision);mpfr_set_prec(hi,precision);oracle(op,lo,x,y,MPFR_RNDD);oracle(op,hi,x,y,MPFR_RNDU);lower=mpfr_get_flt(lo,MPFR_RNDN);upper=mpfr_get_flt(hi,MPFR_RNDN);precision*=2;
   if(precision>16384&&!equal(lower,upper)) {fprintf(stderr,"unresolved %s %08x %08x\n",op,bits(a),bits(b));return 3;}
  } while(!equal(lower,upper));
  uint32_t words[]={bits(a),bits(b),bits(lower)};
  for(int w=0;w<3;w++) for(int j=0;j<4;j++) if(fputc((words[w]>>(8*j))&255,out)==EOF)return 2;
 }
 mpfr_clears(x,y,lo,hi,(mpfr_ptr)0);if(fclose(out))return 2;return 0;
}

/* Independent oracle: integer quotient/remainder rounding and libc fmaf.
 * Normalized x in [1,2). Each LE record: fused MIDP, fused LOWP, RN14, RN8.
 * Bounded output buffer, never an input-space-sized array.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
static uint32_t bits(float f) { uint32_t u;memcpy(&u,&f,4);return u; }
static float value(uint32_t u) { float f;memcpy(&f,&u,4);return f; }
static uint32_t rounded(uint32_t input,unsigned p) {
    uint64_t m=(input&0x7fffffu)|0x800000u, n=UINT64_C(1)<<(23+p);
    uint64_t q=n/m, r=n%m;
    if(2*r>m || (2*r==m && (q&1))) ++q;
    return 0x3f000000u+(uint32_t)((q-(UINT64_C(1)<<(p-1)))<<(24-p));
}
int main(int argc,char **argv) {
    if(argc!=2){fputs("explicit output file required\n",stderr);return 2;}
    if(fesetround(FE_TONEAREST)){return 3;}
    int audit=!strcmp(argv[1],"--audit");
    FILE *f=audit?NULL:fopen(argv[1],"wb");if(!audit && !f){perror(argv[1]);return 1;}
    double pmin[2]={INFINITY,INFINITY},pmax[2]={0,0};
    unsigned char block[16384];uint64_t double_round[2]={0,0};uint32_t witness[2]={0,0};
    for(uint32_t start=0x3f800000u;start<0x40000000u;start+=1024) {
        for(unsigned i=0;i<1024;i++) {
            uint32_t in=start+i;float x=value(in),y=value(0x7ef127eau-in);
            double p0=(double)x*(double)y;
            if(p0<pmin[0])pmin[0]=p0;if(p0>pmax[0])pmax[0]=p0;
            if(!(p0>=0.5 && p0<=2.0)){fputs("seed product outside exact-widening proof\n",stderr);return 1;}
            y=y*fmaf(-x,y,2.0f);uint32_t low=bits(y);
            double p1=(double)x*(double)y;
            if(p1<pmin[1])pmin[1]=p1;if(p1>pmax[1])pmax[1]=p1;
            if(!(p1>=0.5 && p1<=2.0)){fputs("refined product outside exact-widening proof\n",stderr);return 1;}
            y=y*fmaf(-x,y,2.0f);
            uint32_t words[4]={bits(y),low,rounded(in,14),rounded(in,8)};
            for(unsigned j=0;j<2;j++) {
                unsigned drop=24-(j?8:14);uint32_t u=bits(1.0f/x);
                uint32_t naive=(u+(1u<<(drop-1))-1+((u>>drop)&1))&~((1u<<drop)-1);
                if(naive!=words[j+2]){++double_round[j];witness[j]=in;}
            }
            for(unsigned w=0;w<4;w++)for(unsigned b=0;b<4;b++)block[i*16+w*4+b]=(unsigned char)(words[w]>>(b*8));
        }
        if(f && fwrite(block,1,sizeof(block),f)!=sizeof(block)){perror("write");return 1;}
    }
    if(f && fclose(f))return 1;
    printf("Exact-widening product ranges: initial=[%a,%a] refined=[%a,%a]\n",pmin[0],pmax[0],pmin[1],pmax[1]);
    printf("8388608 normalized inputs; 128 MiB LE oracle; fixed 16384-byte buffer\n");
    printf("Naive division-then-quantization mismatches: RN14=%llu witness=%08x; RN8=%llu witness=%08x\n",(unsigned long long)double_round[0],witness[0],(unsigned long long)double_round[1],witness[1]);
    return 0;
}

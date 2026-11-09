// Native ABI benchmark worker. No clocks: zenbench times the parent request.
// Each provider is resolved from its own DSO; no process-wide interposition.
#include <array>
#include <algorithm>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#ifdef __x86_64__
#include <immintrin.h>
#endif
static float from_bits(uint32_t v) {float f; std::memcpy(&f,&v,4);return f;}
static uint32_t bits(float f) {uint32_t v;std::memcpy(&v,&f,4);return v;}
static void *required_dso(const char *name) {
    void *p=dlopen(name,RTLD_NOW|RTLD_LOCAL); if(!p)throw std::runtime_error(dlerror());return p;
}
static void *symbol(void *lib,const char *name) {
    dlerror();void *p=dlsym(lib,name);const char *error=dlerror();if(error)throw std::runtime_error(error);return p;
}
using Scalar=float(*)(float,float);
struct Provider {std::string name;void *fn;int lanes;};
static std::vector<Provider> providers;
#ifdef __x86_64__
__attribute__((target("sse2"))) static void vector4(void *fn,const float *x,const float *n,float *out) {
    using F=__m128(*)(__m128,__m128);
    _mm_storeu_ps(out,reinterpret_cast<F>(fn)(_mm_loadu_ps(x),_mm_loadu_ps(n)));
}
__attribute__((target("avx2,fma"))) static void vector8(void *fn,const float *x,const float *n,float *out) {
    using F=__m256(*)(__m256,__m256);
    _mm256_storeu_ps(out,reinterpret_cast<F>(fn)(_mm256_loadu_ps(x),_mm256_loadu_ps(n)));
}
#endif
static void evaluate(const Provider &p,const float *x,const float *n,float *out,size_t width) {
    if(width%p.lanes)throw std::runtime_error("logical width cannot contain native ABI width");
    for(size_t i=0;i<width;i+=p.lanes) {
        if(p.lanes==1)out[i]=reinterpret_cast<Scalar>(p.fn)(x[i],n[i]);
#ifdef __x86_64__
        else if(p.lanes==4)vector4(p.fn,x+i,n+i,out+i);
        else if(p.lanes==8)vector8(p.fn,x+i,n+i,out+i);
#endif
        else throw std::runtime_error("unsupported ABI width");
    }
}
static void load() {
    if(const char *arm=std::getenv("POWF_ARM_DSO"))providers.push_back({"Arm_v26_07_scalar",symbol(required_dso(arm),"powf"),1});
#ifdef __x86_64__
    auto *gnu=required_dso("libmvec.so.1");
    providers.push_back({"glibc_2_43_libmvec_SSE2x4",symbol(gnu,"_ZGVbN4vv_powf"),4});
    if(__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma"))providers.push_back({"glibc_2_43_libmvec_AVX2x8",symbol(gnu,"_ZGVdN8vv_powf"),8});
    if(const char *aocl=std::getenv("POWF_AOCL_DSO")) {
        auto *lib=required_dso(aocl);
        providers.push_back({"AOCL_5_3_2_scalar",symbol(lib,"amd_powf"),1});
        if(__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma")) {
            providers.push_back({"AOCL_5_3_2_vector4",symbol(lib,"amd_vrs4_powf"),4});
            providers.push_back({"AOCL_5_3_2_vector8",symbol(lib,"amd_vrs8_powf"),8});
        }
    }
#endif
    if(std::fegetround()!=FE_TONEAREST)throw std::runtime_error("requires RNE");
#ifdef __x86_64__
    auto csr=_mm_getcsr(); if(csr&((1u<<15)|(1u<<6)))throw std::runtime_error("requires gradual underflow");
    std::cerr<<"MXCSR="<<std::hex<<csr<<std::dec<<"\n";
#elif defined(__aarch64__)
    uint64_t fpcr;asm volatile("mrs %0, fpcr":"=r"(fpcr));
    if(fpcr&((1ull<<24)|(1ull<<19)))throw std::runtime_error("requires gradual underflow");
    std::cerr<<"FPCR="<<std::hex<<fpcr<<std::dec<<"\n";
#endif
    for(auto &p:providers)std::cerr<<"PROVIDER "<<p.name<<" lanes="<<p.lanes<<"\n";
}
static std::string request(const std::string &line) {
    if(line=="EMPTY")return "00000000\n";
    std::istringstream in(line);std::string op,method;size_t width;in>>op>>method>>width;
    if(!in||!(width==1||width==4||width==8||width==16))throw std::runtime_error("bad width");
    const Provider *p=nullptr;for(auto &v:providers)if(v.name==method)p=&v;
    if(!p)throw std::runtime_error("unknown provider: "+method);
    std::array<float,16>x{},n{},out{};
    if(op=="E") {
        for(size_t i=0;i<width;i++) {uint32_t xb,nb;in>>std::hex>>xb>>nb;x[i]=from_bits(xb);n[i]=from_bits(nb);}
        if(!in)throw std::runtime_error("truncated E input");
        evaluate(*p,x.data(),n.data(),out.data(),width);
    } else if(op=="B") {
        size_t size,repeats;int chain;uint32_t exponent;in>>size>>chain>>std::hex>>exponent>>std::dec>>repeats;
        if(!in||!(size==64||size==4096)||size*repeats!=1048576||(chain!=0&&chain!=1))throw std::runtime_error("bad B workload");
        std::vector<float>input(size),output(size);
        for(size_t i=0;i<size;i++)input[i]=float((i%255)+1)/255.f;
        n.fill(from_bits(exponent));
        for(size_t r=0;r<repeats;r++) {
            std::copy_n(input.begin(),width,x.begin());
            for(size_t offset=0;offset<size;offset+=width) {
                if(!chain)std::copy_n(input.begin()+offset,width,x.begin());
                evaluate(*p,x.data(),n.data(),out.data(),width);
                for(size_t i=0;i<width;i++) {
                    x[i]=chain?from_bits(0x3f000000|(bits(out[i])&0x007fffff)):out[i];output[offset+i]=x[i];
                }
            }
            asm volatile(""::"m"(output):"memory");
        }
        std::copy_n(output.end()-width,width,out.begin());
    } else throw std::runtime_error("bad operation");
    std::ostringstream response;response<<std::hex;
    for(size_t i=0;i<width;i++){response.width(8);response.fill('0');response<<bits(out[i])<<' ';}
    response<<'\n';return response.str();
}
static void connection(int fd) {
    FILE *file=fdopen(fd,"r+"); if(!file)std::abort();
    char *line=nullptr;size_t capacity=0;ssize_t length;
    while((length=getline(&line,&capacity,file))>0) {
        std::string text(line,size_t(length));if(text.back()=='\n')text.pop_back();
        auto result=request(text); if(fwrite(result.data(),1,result.size(),file)!=result.size())std::abort();fflush(file);
    }
    free(line);fclose(file);
}
int main(int argc,char **argv) {
    try {
        load();
        if(argc==1) {std::string line;while(std::getline(std::cin,line))std::cout<<request(line)<<std::flush;return 0;}
        if(argc!=2||std::strlen(argv[1])>=sizeof(sockaddr_un::sun_path))throw std::runtime_error("socket path too long");
        int fd=socket(AF_UNIX,SOCK_STREAM,0);if(fd<0)throw std::runtime_error("socket");
        sockaddr_un address{};address.sun_family=AF_UNIX;std::strcpy(address.sun_path,argv[1]);
        if(bind(fd,reinterpret_cast<sockaddr*>(&address),sizeof(address))||listen(fd,32))throw std::runtime_error("bind/listen");
        for(;;){int client=accept(fd,nullptr,nullptr);if(client<0)throw std::runtime_error("accept");std::thread(connection,client).detach();}
    } catch(const std::exception &e){std::cerr<<"FAILED: "<<e.what()<<'\n';return 1;}
}

#define _GNU_SOURCE
#include <dlfcn.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

/* Inspect the resolved IFUNC entry; never infer its ISA from the public ABI. */
int main(void) {
    void *library = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
    if (!library) { fputs(dlerror(), stderr); return 1; }
    void *function = dlvsym(library, "powf", "GLIBC_2.27");
    Dl_info info;
    if (!function || !dladdr(function, &info)) return 2;
    printf("powf@GLIBC_2.27 %s offset=0x%" PRIxPTR "\n", info.dli_fname,
           (uintptr_t)function - (uintptr_t)info.dli_fbase);
    return 0;
}

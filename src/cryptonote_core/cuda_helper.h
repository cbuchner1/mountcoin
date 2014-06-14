#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

static __device__ unsigned long long MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
#if __CUDA_ARCH__ >= 130
    return __double_as_longlong(__hiloint2double(HI, LO));
#else
	return (unsigned long long)LO | (((unsigned long long)HI) << 32);
#endif
}

// das Hi Word aus einem 64 Bit Typen extrahieren
static __device__ uint32_t HIWORD(const unsigned long long &x) {
#if __CUDA_ARCH__ >= 130
	return (uint32_t)__double2hiint(__longlong_as_double(x));
#else
	return (uint32_t)(x >> 32);
#endif
}

// das Hi Word in einem 64 Bit Typen ersetzen
static __device__ unsigned long long REPLACE_HIWORD(const unsigned long long &x, const uint32_t &y) {
	return (x & 0xFFFFFFFFULL) | (((unsigned long long)y) << 32ULL);
}

// das Lo Word aus einem 64 Bit Typen extrahieren
static __device__ uint32_t LOWORD(const unsigned long long &x) {
#if __CUDA_ARCH__ >= 130
	return (uint32_t)__double2loint(__longlong_as_double(x));
#else
	return (uint32_t)(x & 0xFFFFFFFFULL);
#endif
}

// das Lo Word in einem 64 Bit Typen ersetzen
static __device__ unsigned long long REPLACE_LOWORD(const unsigned long long &x, const uint32_t &y) {
	return (x & 0xFFFFFFFF00000000ULL) | ((unsigned long long)y);
}

// Endian Drehung f�r 32 Bit Typen
static __device__ uint32_t cuda_swab32(uint32_t x)
{
	return __byte_perm(x, x, 0x0123);
}

// Endian Drehung f�r 64 Bit Typen
static __device__ unsigned long long cuda_swab64(unsigned long long x) {
    return MAKE_ULONGLONG(cuda_swab32(HIWORD(x)), cuda_swab32(LOWORD(x)));
}

// diese 64 Bit Rotates werden unter Compute 3.5 (und besser) mit dem Funnel Shifter beschleunigt
#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ unsigned long long ROTR64(const unsigned long long value, const int offset) {
    uint2 result;
    if(offset < 32) {
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
    } else {
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
    }
    return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#else
#define ROTR64(x, n)        (((x) >> (n)) | ((x) << (64 - (n))))
#endif

// diese 64 Bit Rotates werden unter Compute 3.5 (und besser) mit dem Funnel Shifter beschleunigt
#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ unsigned long long ROTL64(const unsigned long long value, const int offset) {
    uint2 result;
    if(offset >= 32) {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
    } else {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
    }
    return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#else
#define ROTL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))
#endif

#endif // #ifndef CUDA_HELPER_H

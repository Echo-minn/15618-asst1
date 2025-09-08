#include <smmintrin.h> // For _mm_stream_load_si128
#include <emmintrin.h> // For _mm_mul_ps
#include <xmmintrin.h> // For _mm_stream_ps
#include <assert.h>
#include <stdint.h>

extern void saxpySerial(int N,
			float scale,
			float X[],
			float Y[],
			float result[]);


void saxpyStreaming(int N,
                    float scale,
                    float X[],
                    float Y[],
                    float result[])
{
    // Use Intel intrinsics with non-temporal memory hints to reduce memory traffic
    // This implementation reduces memory footprint to 3*N*sizeof(float) by using
    // streaming stores that bypass the cache for the result array
    
    int i;
    __m128 vecScale = _mm_set1_ps(scale); // Broadcast scale to all 4 elements
    
    // Process 4 floats at a time using SIMD
    for (i = 0; i <= N - 4; i += 4) {
        // Load 4 elements from X and Y arrays
        __m128 vecX = _mm_loadu_ps(&X[i]);
        __m128 vecY = _mm_loadu_ps(&Y[i]);
        
        // Perform SAXPY operation: scale * X[i] + Y[i]
        __m128 vecResult = _mm_add_ps(_mm_mul_ps(vecScale, vecX), vecY);
        
        // Store result directly to memory using non-temporal store (bypasses cache)
        _mm_stream_ps(&result[i], vecResult);
    }
    
    // if N is not divisible by 4
    for (; i < N; i++) {
        result[i] = scale * X[i] + Y[i];
    }
    
    // Ensure all streaming stores are completed before function returns
    _mm_sfence();
}


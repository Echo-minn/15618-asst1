#include <algorithm>

// Generate random data
void initRandom(float *values, int N) {
    for (int i=0; i<N; i++)
    {
        // random input values
        values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
    }
}

// Generate data that gives high relative speedup
void initGood(float *values, int N) {
    for (int i=0; i<N; i++)
    {
        // Todo: Choose values
        values[i] = 2.999f - i / N;
    }
}

// Generate data that gives low relative speedup
void initBad(float *values, int N) {
    for (int i=0; i<N; i++)
    {
        // ispc vector work on 8 lanes, make them have highest variance can make them less efficient
        // This creates greatest load imbalance within SIMD lanes
        if (i % 8 == 0) {
            values[i] = 2.999f;
        } else {
            values[i] = 1.0f;    // Cheap computation, all wait for the first lane to finish
        }
    }
}


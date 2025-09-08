
export void saxpy_ispc(uniform int N,
                       uniform float scale,
                            uniform float X[],
                            uniform float Y[],
                            uniform float result[])
{
    foreach (i = 0 ... N) {           
        result[i] = scale * X[i] + Y[i];
    }
}

// Cache-friendly blocked version with prefetching
task void saxpy_ispc_blocked_task(uniform int N,
                                 uniform int taskStart,
                                 uniform int taskEnd,
                                 uniform float scale,
                                 uniform float X[], 
                                 uniform float Y[],
                                 uniform float result[])
{
    // Use smaller blocks that fit in L2 cache (256KB typical)
    // Each block: 3 arrays * 4 bytes * block_size = 12 * block_size
    // For 64KB per block: block_size = 64KB / 12 ≈ 5461 elements
    uniform int BLOCK_SIZE = 5461;
    
    // Process assigned range in cache-friendly blocks
    for (uniform int blockStart = taskStart; blockStart < taskEnd; blockStart += BLOCK_SIZE) {
        uniform int blockEnd = min(taskEnd, blockStart + BLOCK_SIZE);
        
        // Prefetch next block while processing current block
        if (blockEnd < taskEnd) {
            uniform int prefetchStart = blockEnd;
            uniform int prefetchEnd = min(taskEnd, prefetchStart + BLOCK_SIZE);
            
            // Prefetch next block data
            foreach (i = prefetchStart ... min(prefetchEnd, prefetchStart + 64)) {
                prefetch_l1(&X[i]);
                prefetch_l1(&Y[i]);
            }
        }
        
        // Process current block with SIMD
        foreach (i = blockStart ... blockEnd) {
            result[i] = scale * X[i] + Y[i];
        }
    }
}

// Enhanced version with multiple passes to increase arithmetic intensity
task void saxpy_ispc_multipass_task(uniform int N,
                                   uniform int taskStart,
                                   uniform int taskEnd,
                                   uniform float scale,
                                   uniform float X[], 
                                   uniform float Y[],
                                   uniform float result[])
{
    uniform int BLOCK_SIZE = 5461;
    uniform int NUM_PASSES = 4; // Multiple passes to increase arithmetic intensity
    
    for (uniform int blockStart = taskStart; blockStart < taskEnd; blockStart += BLOCK_SIZE) {
        uniform int blockEnd = min(taskEnd, blockStart + BLOCK_SIZE);
        
        // Multiple passes over the same data to increase arithmetic intensity
        foreach (i = blockStart ... blockEnd) {
            // First pass: standard SAXPY
            result[i] = scale * X[i] + Y[i];
            
            // Additional passes: reuse loaded data for more computation
            for (uniform int pass = 1; pass < NUM_PASSES; pass++) {
                result[i] = scale * result[i] + Y[i];
            }
        }
    }
}

task void saxpy_ispc_task(uniform int N,
                               uniform int span,
                               uniform float scale,
                               uniform float X[], 
                               uniform float Y[],
                               uniform float result[])
{

    uniform int indexStart = taskIndex * span;
    uniform int indexEnd = min(N, indexStart + span);

    foreach (i = indexStart ... indexEnd) {
        result[i] = scale * X[i] + Y[i];
    }
}

export void saxpy_ispc_withtasks(uniform int N,
                               uniform float scale,
                               uniform float X[],
                               uniform float Y[],
                               uniform float result[])
{

    uniform int span = N / 64;  // 64 tasks

    launch[N/span] saxpy_ispc_task(N, span, scale, X, Y, result);
}

// Task wrapper for blocked implementation
task void saxpy_ispc_blocked_wrapper(uniform int N,
                                    uniform int elementsPerTask,
                                    uniform float scale,
                                    uniform float X[], 
                                    uniform float Y[],
                                    uniform float result[])
{
    uniform int taskStart = taskIndex * elementsPerTask;
    uniform int taskEnd = min(N, taskStart + elementsPerTask);
    
    saxpy_ispc_blocked_task(N, taskStart, taskEnd, scale, X, Y, result);
}

// Task wrapper for multipass implementation
task void saxpy_ispc_multipass_wrapper(uniform int N,
                                      uniform int elementsPerTask,
                                      uniform float scale,
                                      uniform float X[], 
                                      uniform float Y[],
                                      uniform float result[])
{
    uniform int taskStart = taskIndex * elementsPerTask;
    uniform int taskEnd = min(N, taskStart + elementsPerTask);
    
    saxpy_ispc_multipass_task(N, taskStart, taskEnd, scale, X, Y, result);
}

// Optimized version with cache blocking and better task distribution
export void saxpy_ispc_blocked(uniform int N,
                              uniform float scale,
                              uniform float X[],
                              uniform float Y[],
                              uniform float result[])
{
    // Use fewer tasks but larger work units for better cache utilization
    // Typically 2x number of CPU cores for good load balancing
    uniform int NUM_TASKS = 16;
    uniform int elementsPerTask = (N + NUM_TASKS - 1) / NUM_TASKS;
    
    launch[NUM_TASKS] saxpy_ispc_blocked_wrapper(N, elementsPerTask, scale, X, Y, result);
}

// Version with increased arithmetic intensity
export void saxpy_ispc_multipass(uniform int N,
                                 uniform float scale,
                                 uniform float X[],
                                 uniform float Y[],
                                 uniform float result[])
{
    uniform int NUM_TASKS = 16;
    uniform int elementsPerTask = (N + NUM_TASKS - 1) / NUM_TASKS;
    
    launch[NUM_TASKS] saxpy_ispc_multipass_wrapper(N, elementsPerTask, scale, X, Y, result);
}

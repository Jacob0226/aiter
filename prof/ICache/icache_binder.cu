// icache_binder.cu

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void flush_icache_kernel()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}

void flush_icache() {
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    int n_processor = prop->multiProcessorCount;
    dim3 grid(n_processor * 60);
    dim3 block(64);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    printf("Run flush_icache(). n_processor=%d\n", n_processor);
    flush_icache_kernel<<<grid, block, 0, stream>>>();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)
        );
    }
}


namespace py = pybind11;
PYBIND11_MODULE(icache_module, m) {
    m.doc() = "A module to flush GPU Instruction Cache."; 
    m.def("flush_icache", &flush_icache, "Flush the GPU Instruction Cache across all streaming multiprocessors.");
}
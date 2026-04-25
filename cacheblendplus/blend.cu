#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * CUDA kernel for blending KV caches.
 * 
 * cached_kv: (L, 2, N, H, D)
 * new_values: (L, 2, k, H, D)
 * indices: (k,)
 * 
 * Each thread handles one element in the (L, 2, k, H, D) space.
 */
__global__ void blend_kernel(
    at::Half* __restrict__ cached_kv,
    const at::Half* __restrict__ new_values,
    const int64_t* __restrict__ indices,
    int L, int N, int H, int D, int k
) {
    // Thread index in the output space (L, 2, k, H, D)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = L * 2 * k * H * D;

    if (tid < total_elements) {
        // Decompose tid into (l, kv, ki, h, d)
        int d = tid % D;
        int tmp = tid / D;
        int h = tmp % H;
        tmp /= H;
        int ki = tmp % k;
        tmp /= k;
        int kv = tmp % 2;
        int l = tmp / 2;

        // Get the target index in the original cache
        int64_t target_idx = indices[ki];

        // Compute offsets
        // cached_kv shape: (L, 2, N, H, D)
        int64_t cached_offset = (((int64_t)l * 2 + kv) * N + target_idx) * H * D + h * D + d;
        
        // new_values shape: (L, 2, k, H, D)
        int64_t new_offset = tid;

        cached_kv[cached_offset] = new_values[new_offset];
    }
}

void launch_blend(
    at::Tensor cached_kv,
    at::Tensor new_values,
    at::Tensor indices
) {
    int L = cached_kv.size(0);
    int N = cached_kv.size(2);
    int H = cached_kv.size(3);
    int D = cached_kv.size(4);
    int k = new_values.size(2);

    int total_elements = L * 2 * k * H * D;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    blend_kernel<<<blocks, threads_per_block>>>(
        (at::Half*)cached_kv.data_ptr(),
        (const at::Half*)new_values.data_ptr(),
        (const int64_t*)indices.data_ptr(),
        L, N, H, D, k
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_blend", &launch_blend, "Launch the CUDA blend kernel");
}

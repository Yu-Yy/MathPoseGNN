// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
__global__ void gather_points_kernel(int b, int c, int n, int m,
                                     const float *__restrict__ points,
                                     const int *__restrict__ idx,
                                     float *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out) {
  gather_points_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                         at::cuda::getCurrentCUDAStream()>>>(b, c, n, npoints,
                                                             points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
                                          const float *__restrict__ grad_out,
                                          const int *__restrict__ idx,
                                          float *__restrict__ grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        atomicAdd(grad_points + (i * c + l) * n + a,
                  grad_out[(i * c + l) * m + j]);
      }
    }
  }
}

void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points) {
  gather_points_grad_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, npoints, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
  if (m <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * 21;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float u1 = dataset[old * 21 + 0];
    float u2 = dataset[old * 21 + 1];
    float u3 = dataset[old * 21 + 2];
    float C11 = dataset[old * 21 + 3];
    float C12 = dataset[old * 21 + 4];
    float C13 = dataset[old * 21 + 5];
    float C21 = dataset[old * 21 + 6];
    float C22 = dataset[old * 21 + 7];
    float C23 = dataset[old * 21 + 8];
    float C31 = dataset[old * 21 + 9];
    float C32 = dataset[old * 21 + 10];
    float C33 = dataset[old * 21 + 11];
    float D11 = dataset[old * 21 + 12];
    float D12 = dataset[old * 21 + 13];
    float D13 = dataset[old * 21 + 14];
    float D21 = dataset[old * 21 + 15];
    float D22 = dataset[old * 21 + 16];
    float D23 = dataset[old * 21 + 17];
    float D31 = dataset[old * 21 + 18];
    float D32 = dataset[old * 21 + 19];
    float D33 = dataset[old * 21 + 20];

    for (int k = tid; k < n; k += stride) {
      float u1_ = dataset[k * 21 + 0];
      float u2_ = dataset[k * 21 + 1];
      float u3_ = dataset[k * 21 + 2];
      float C11_ = dataset[k * 21 + 3];
      float C12_ = dataset[k * 21 + 4];
      float C13_ = dataset[k * 21 + 5];
      float C21_ = dataset[k * 21 + 6];
      float C22_ = dataset[k * 21 + 7];
      float C23_ = dataset[k * 21 + 8];
      float C31_ = dataset[k * 21 + 9];
      float C32_ = dataset[k * 21 + 10];
      float C33_ = dataset[k * 21 + 11];
      float D11_ = dataset[k * 21 + 12];
      float D12_ = dataset[k * 21 + 13];
      float D13_ = dataset[k * 21 + 14];
      float D21_ = dataset[k * 21 + 15];
      float D22_ = dataset[k * 21 + 16];
      float D23_ = dataset[k * 21 + 17];
      float D31_ = dataset[k * 21 + 18];
      float D32_ = dataset[k * 21 + 19];
      float D33_ = dataset[k * 21 + 20];
      
      float mag = (u1_ * u1_) + (u2_ * u2_) + (u3_ * u3_);
      if (mag <= 1e-3) continue;

      float d =
          ((u1 - u1_)*(D11*(u1 - u1_) + D21*(u2 - u2_) + D31*(u3 - u3_)) + 
          (u2 - u2_)*(D12*(u1 - u1_) + D22*(u2 - u2_) + D32*(u3 - u3_)) + 
          (u3 - u3_)*(D13*(u1 - u1_) + D23*(u2 - u2_) + D33*(u3 - u3_)) + 
          (u1 - u1_)*(D11_*(u1 - u1_) + D21_*(u2 - u2_) + D31_*(u3 - u3_)) + 
          (u2 - u2_)*(D12_*(u1 - u1_) + D22_*(u2 - u2_) + D32_*(u3 - u3_)) + 
          (u3 - u3_)*(D13_*(u1 - u1_) + D23_*(u2 - u2_) + D33_*(u3 - u3_)) + 
          C11*D11_ + C11_*D11 + C12*D21_ + C21*D12_ + C12_*D21 + C21_*D12 + C13*D31_ + 
          C22*D22_ + C31*D13_ + C13_*D31 + C22_*D22 + C31_*D13 + C23*D32_ + C32*D23_ + 
          C23_*D32 + C32_*D23 + C33*D33_ + C33_*D33 - 6) / 4;

      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 512) {
      if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs) {
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
    case 512:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_kernel<256>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_kernel<128>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_kernel<64>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_kernel<32>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_kernel<16>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_kernel<8>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_kernel<4>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_kernel<2>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 1:
      furthest_point_sampling_kernel<1>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
  }

  CUDA_CHECK_ERRORS();
}

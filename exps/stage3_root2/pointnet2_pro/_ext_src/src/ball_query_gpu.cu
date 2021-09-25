// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 21;
  new_xyz += batch_index * m * 21;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    // float new_x = new_xyz[j * 3 + 0];
    // float new_y = new_xyz[j * 3 + 1];
    // float new_z = new_xyz[j * 3 + 2];

    float u1 = new_xyz[j * 21 + 0];
    float u2 = new_xyz[j * 21 + 1];
    float u3 = new_xyz[j * 21 + 2];
    float C11 = new_xyz[j * 21 + 3];
    float C12 = new_xyz[j * 21 + 4];
    float C13 = new_xyz[j * 21 + 5];
    float C21 = new_xyz[j * 21 + 6];
    float C22 = new_xyz[j * 21 + 7];
    float C23 = new_xyz[j * 21 + 8];
    float C31 = new_xyz[j * 21 + 9];
    float C32 = new_xyz[j * 21 + 10];
    float C33 = new_xyz[j * 21 + 11];
    float D11 = new_xyz[j * 21 + 12];
    float D12 = new_xyz[j * 21 + 13];
    float D13 = new_xyz[j * 21 + 14];
    float D21 = new_xyz[j * 21 + 15];
    float D22 = new_xyz[j * 21 + 16];
    float D23 = new_xyz[j * 21 + 17];
    float D31 = new_xyz[j * 21 + 18];
    float D32 = new_xyz[j * 21 + 19];
    float D33 = new_xyz[j * 21 + 20];

    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      // float x = xyz[k * 3 + 0];
      // float y = xyz[k * 3 + 1];
      // float z = xyz[k * 3 + 2];

      float u1_ = xyz[k * 21 + 0];
      float u2_ = xyz[k * 21 + 1];
      float u3_ = xyz[k * 21 + 2];
      float C11_ = xyz[k * 21 + 3];
      float C12_ = xyz[k * 21 + 4];
      float C13_ = xyz[k * 21 + 5];
      float C21_ = xyz[k * 21 + 6];
      float C22_ = xyz[k * 21 + 7];
      float C23_ = xyz[k * 21 + 8];
      float C31_ = xyz[k * 21 + 9];
      float C32_ = xyz[k * 21 + 10];
      float C33_ = xyz[k * 21 + 11];
      float D11_ = xyz[k * 21 + 12];
      float D12_ = xyz[k * 21 + 13];
      float D13_ = xyz[k * 21 + 14];
      float D21_ = xyz[k * 21 + 15];
      float D22_ = xyz[k * 21 + 16];
      float D23_ = xyz[k * 21 + 17];
      float D31_ = xyz[k * 21 + 18];
      float D32_ = xyz[k * 21 + 19];
      float D33_ = xyz[k * 21 + 20];

      float d2 = ((u1 - u1_)*(D11*(u1 - u1_) + D21*(u2 - u2_) + D31*(u3 - u3_)) + 
          (u2 - u2_)*(D12*(u1 - u1_) + D22*(u2 - u2_) + D32*(u3 - u3_)) + 
          (u3 - u3_)*(D13*(u1 - u1_) + D23*(u2 - u2_) + D33*(u3 - u3_)) + 
          (u1 - u1_)*(D11_*(u1 - u1_) + D21_*(u2 - u2_) + D31_*(u3 - u3_)) + 
          (u2 - u2_)*(D12_*(u1 - u1_) + D22_*(u2 - u2_) + D32_*(u3 - u3_)) + 
          (u3 - u3_)*(D13_*(u1 - u1_) + D23_*(u2 - u2_) + D33_*(u3 - u3_)) + 
          C11*D11_ + C11_*D11 + C12*D21_ + C21*D12_ + C12_*D21 + C21_*D12 + C13*D31_ + 
          C22*D22_ + C31*D13_ + C13_*D31 + C22_*D22 + C31_*D13 + C23*D32_ + C32*D23_ + 
          C23_*D32 + C32_*D23 + C33*D33_ + C33_*D33 - 6) / 4;

      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_xyz, xyz, idx);

  CUDA_CHECK_ERRORS();
}

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file roi_pooling.cu
 * \brief roi pooling operator
 * \author Ross Girshick, Kye-Hyeon Kim, Jian Guo
*/
#include "./quadrangle_roi_pooling-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include <cstdio>

namespace mshadow {
namespace cuda {

template<typename Dtype>
__global__ void QROIPoolForwardKernel(const int count, const Dtype* bottom_data,
                                     const float spatial_scale, const int channels,
                                     const int height, const int width,
                                     const int pooled_height, const int pooled_width,
                                     const Dtype* bottom_rois, Dtype* top_data,
                                     Dtype* argmax_data) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 9;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      argmax_data[index] = 0;
      continue;
    }

    Dtype P[8];
    P[0] = bottom_rois[1] * spatial_scale;
    P[1] = bottom_rois[2] * spatial_scale;
    P[2] = bottom_rois[3] * spatial_scale;
    P[3] = bottom_rois[4] * spatial_scale;
    P[4] = bottom_rois[5] * spatial_scale;
    P[5] = bottom_rois[6] * spatial_scale;
    P[6] = bottom_rois[7] * spatial_scale;
    P[7] = bottom_rois[8] * spatial_scale;

    // horizon bbox in feature map
    int roi_start_w = round(fmin(fmin(P[0],P[2]),fmin(P[4],P[6])));
    int roi_start_h = round(fmin(fmin(P[1],P[3]),fmin(P[5],P[7])));
    int roi_end_w = round(fmax(fmax(P[0],P[2]),fmax(P[4],P[6])));
    int roi_end_h = round(fmax(fmax(P[1],P[3]),fmax(P[5],P[7])));

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        // check if point (w,h) in polygon
        bool isInside = false;
        int pn = 4; // point num
        for (int i = 0, j = pn - 1; i < pn; j = i++) {
             Dtype ix = P[i * 2], iy = P[i * 2 + 1];
             Dtype jx = P[j * 2], jy = P[j * 2 + 1];
             if ( (iy > h) != (jy > h) &&
                    w < (jx - ix) * (h - iy) / (jy - iy) + ix ) {
                isInside = !isInside;
             }
        }
        if(!isInside) continue;
        int bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    if (maxidx == -1) maxval = 0;
    top_data[index] = maxval;
    argmax_data[index] = (Dtype)maxidx;
    //if (maxidx == -1) printf("bad: %d\n", maxval);
    //else printf("good: %d\n", maxval);
  }
}

template<typename Dtype>
inline void QROIPoolForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "QROIPooling Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  QROIPoolForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, top_data, argmax_data);
  MSHADOW_CUDA_POST_KERNEL_CHECK(QROIPoolForwardKernel);
}

template<typename Dtype>
__global__ void QROIPoolBackwardAccKernel(const int count, const Dtype* top_diff,
                                         const Dtype* argmax_data, const int num_rois,
                                         const float spatial_scale, const int channels,
                                         const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         Dtype* bottom_diff, const Dtype* bottom_rois) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    /* the way of loop top
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    
    bottom_rois += n * 9; // box dim is 9
    int roi_batch_ind = bottom_rois[0];
    bottom_diff += (roi_batch_ind * channels + c) * height * width;

    int bottom_index = argmax_data[index];
    if(bottom_index != -1) bottom_diff[bottom_index] += top_diff[index];
    */

    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 9;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      Dtype P[8];
      P[0] = offset_bottom_rois[1] * spatial_scale;
      P[1] = offset_bottom_rois[2] * spatial_scale;
      P[2] = offset_bottom_rois[3] * spatial_scale;
      P[3] = offset_bottom_rois[4] * spatial_scale;
      P[4] = offset_bottom_rois[5] * spatial_scale;
      P[5] = offset_bottom_rois[6] * spatial_scale;
      P[6] = offset_bottom_rois[7] * spatial_scale;
      P[7] = offset_bottom_rois[8] * spatial_scale;

      // horizon bbox in feature map
      int roi_start_w = round(fmin(fmin(P[0],P[2]),fmin(P[4],P[6])));
      int roi_start_h = round(fmin(fmin(P[1],P[3]),fmin(P[5],P[7])));
      int roi_end_w = round(fmax(fmax(P[0],P[2]),fmax(P[4],P[6])));
      int roi_end_h = round(fmax(fmax(P[1],P[3]),fmax(P[5],P[7])));

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      // check if point (w,h) in polygon
      bool isInside = false;
      int pn = 4; // point num
      for (int i = 0, j = pn - 1; i < pn; j = i++) {
           Dtype ix = P[i * 2], iy = P[i * 2 + 1];
           Dtype jx = P[j * 2], jy = P[j * 2 + 1];
           if ( (iy > h) != (jy > h) &&
                  w < (jx - ix) * (h - iy) / (jy - iy) + ix ) {
              isInside = !isInside;
           }
      }
      if(!isInside) continue;

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (static_cast<int>(offset_argmax_data[ph * pooled_width + pw]) == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] += gradient;
  }
}

template<typename Dtype>
inline void QROIPoolBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, Dtype> &max_idx,
                               const float spatial_scale) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int count = in_grad.shape_.Size();
  //fyk: we change the for loop of top instead of bottom
  //const int count = out_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_grad.size(1);
  const int height = in_grad.size(2);
  const int width = in_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "QROIPooling Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  // change to out_grad stream
  //cudaStream_t stream = Stream<gpu>::GetStream(out_grad.stream_);
  QROIPoolBackwardAccKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, argmax_data, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_diff, bottom_rois);
  MSHADOW_CUDA_POST_KERNEL_CHECK(QROIPoolBackwardAccKernel);
}

}  // namespace cuda

template<typename Dtype>
inline void QROIPoolForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  cuda::QROIPoolForward(out, data, bbox, max_idx, spatial_scale);
}

template<typename Dtype>
inline void QROIPoolBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, Dtype> &max_idx,
                               const float spatial_scale) {
  cuda::QROIPoolBackwardAcc(in_grad, out_grad, bbox, max_idx, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(QROIPoolingParam param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QROIPoolingOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

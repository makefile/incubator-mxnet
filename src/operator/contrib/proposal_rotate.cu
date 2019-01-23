/*!
 * Copyright (c) 2015 by Contributors
 * \file proposal.cu
 * \brief Proposal Operator
 * \author Shaoqing Ren, Jian Guo
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <iostream>

#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./proposal_rotate-inl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {

// scores are (b, anchor, h, w)
// workspace_proposals are (h * w * anchor, 5)
// w defines "x" and h defines "y"
// count should be total anchors numbers, h * w * anchors
template<typename Dtype>
__global__ void ProposalGridKernel(const int count,
                                   const int num_anchors,
                                   const int height,
                                   const int width,
                                   const int feature_stride,
                                   const Dtype* scores,
                                   Dtype* workspace_proposals) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % width;
    int h = index / num_anchors / width;

    workspace_proposals[index * 9 + 0] = workspace_proposals[a * 9 + 0] + w * feature_stride;
    workspace_proposals[index * 9 + 1] = workspace_proposals[a * 9 + 1] + h * feature_stride;
    workspace_proposals[index * 9 + 2] = workspace_proposals[a * 9 + 2] + w * feature_stride;
    workspace_proposals[index * 9 + 3] = workspace_proposals[a * 9 + 3] + h * feature_stride;
    workspace_proposals[index * 9 + 4] = workspace_proposals[a * 9 + 4] + w * feature_stride;
    workspace_proposals[index * 9 + 5] = workspace_proposals[a * 9 + 5] + h * feature_stride;
    workspace_proposals[index * 9 + 6] = workspace_proposals[a * 9 + 6] + w * feature_stride;
    workspace_proposals[index * 9 + 7] = workspace_proposals[a * 9 + 7] + h * feature_stride;
    workspace_proposals[index * 9 + 8] = scores[(a * height + h) * width + w];
  }
}

// boxes are (h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (h * w * anchor, 5)
// count should be total anchors numbers, h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename Dtype>
__global__ void BBoxPredKernel(const int count,
                               const int num_anchors,
                               const int feat_height,
                               const int feat_width,
                               const int real_height,
                               const int real_width,
                               const float im_height,
                               const float im_width,
                               const Dtype* boxes,
                               const Dtype* deltas,
                               Dtype* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = index / num_anchors / feat_width;

    float x1 = boxes[index * 9 + 0];
    float y1 = boxes[index * 9 + 1];
    float x2 = boxes[index * 9 + 2];
    float y2 = boxes[index * 9 + 3];
    float x3 = boxes[index * 9 + 4];
    float y3 = boxes[index * 9 + 5];
    float x4 = boxes[index * 9 + 6];
    float y4 = boxes[index * 9 + 7];

    float xmin = fmin(fmin(x1, x2), fmin(x3, x4));
    float ymin = fmin(fmin(y1, y2), fmin(y3, y4));
    float xmax = fmax(fmax(x1, x2), fmax(x3, x4));
    float ymax = fmax(fmax(y1, y2), fmax(y3, y4));
    float width  = xmax - xmin + 1.0f;
    float height = ymax - ymin + 1.0f;

    float dx1 = deltas[((a * 8 + 0) * feat_height + h) * feat_width + w];
    float dy1 = deltas[((a * 8 + 1) * feat_height + h) * feat_width + w];
    float dx2 = deltas[((a * 8 + 2) * feat_height + h) * feat_width + w];
    float dy2 = deltas[((a * 8 + 3) * feat_height + h) * feat_width + w];
    float dx3 = deltas[((a * 8 + 4) * feat_height + h) * feat_width + w];
    float dy3 = deltas[((a * 8 + 5) * feat_height + h) * feat_width + w];
    float dx4 = deltas[((a * 8 + 6) * feat_height + h) * feat_width + w];
    float dy4 = deltas[((a * 8 + 7) * feat_height + h) * feat_width + w];

    float pred_x1 = x1 + dx1 * width;
    float pred_y1 = y1 + dy1 * height;
    float pred_x2 = x2 + dx2 * width;
    float pred_y2 = y2 + dy2 * height;
    float pred_x3 = x3 + dx3 * width;
    float pred_y3 = y3 + dy3 * height;
    float pred_x4 = x4 + dx4 * width;
    float pred_y4 = y4 + dy4 * height;

    pred_x1 = max(min(pred_x1, im_width - 1.0f), 0.0f);
    pred_y1 = max(min(pred_y1, im_height - 1.0f), 0.0f);
    pred_x2 = max(min(pred_x2, im_width - 1.0f), 0.0f);
    pred_y2 = max(min(pred_y2, im_height - 1.0f), 0.0f);
    pred_x3 = max(min(pred_x3, im_width - 1.0f), 0.0f);
    pred_y3 = max(min(pred_y3, im_height - 1.0f), 0.0f);
    pred_x4 = max(min(pred_x4, im_width - 1.0f), 0.0f);
    pred_y4 = max(min(pred_y4, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 9 + 0] = pred_x1;
    out_pred_boxes[index * 9 + 1] = pred_y1;
    out_pred_boxes[index * 9 + 2] = pred_x2;
    out_pred_boxes[index * 9 + 3] = pred_y2;
    out_pred_boxes[index * 9 + 4] = pred_x3;
    out_pred_boxes[index * 9 + 5] = pred_y3;
    out_pred_boxes[index * 9 + 6] = pred_x4;
    out_pred_boxes[index * 9 + 7] = pred_y4;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 9 + 8] = -1.0f;
    }
  }
}

// boxes are (h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (h * w * anchor, 5)
// count should be total anchors numbers, h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename Dtype>
__global__ void IoUPredKernel(const int count,
                              const int num_anchors,
                              const int feat_height,
                              const int feat_width,
                              const int real_height,
                              const int real_width,
                              const float im_height,
                              const float im_width,
                              const Dtype* boxes,
                              const Dtype* deltas,
                              Dtype* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = index / num_anchors / feat_width;

    float x1 = boxes[index * 9 + 0];
    float y1 = boxes[index * 9 + 1];
    float x2 = boxes[index * 9 + 2];
    float y2 = boxes[index * 9 + 3];
    float x3 = boxes[index * 9 + 4];
    float y3 = boxes[index * 9 + 5];
    float x4 = boxes[index * 9 + 6];
    float y4 = boxes[index * 9 + 7];

    float dx1 = deltas[((a * 8 + 0) * feat_height + h) * feat_width + w];
    float dy1 = deltas[((a * 8 + 1) * feat_height + h) * feat_width + w];
    float dx2 = deltas[((a * 8 + 2) * feat_height + h) * feat_width + w];
    float dy2 = deltas[((a * 8 + 3) * feat_height + h) * feat_width + w];
    float dx3 = deltas[((a * 8 + 4) * feat_height + h) * feat_width + w];
    float dy3 = deltas[((a * 8 + 5) * feat_height + h) * feat_width + w];
    float dx4 = deltas[((a * 8 + 6) * feat_height + h) * feat_width + w];
    float dy4 = deltas[((a * 8 + 7) * feat_height + h) * feat_width + w];

    out_pred_boxes[index * 9 + 0] = max(min(x1 + dx1, im_width - 1.0f), 0.0f);
    out_pred_boxes[index * 9 + 1] = max(min(y1 + dy1, im_height - 1.0f), 0.0f);
    out_pred_boxes[index * 9 + 2] = max(min(x2 + dx2, im_width - 1.0f), 0.0f);
    out_pred_boxes[index * 9 + 3] = max(min(y2 + dy2, im_height - 1.0f), 0.0f);
    out_pred_boxes[index * 9 + 4] = max(min(x3 + dx3, im_width - 1.0f), 0.0f);
    out_pred_boxes[index * 9 + 5] = max(min(y3 + dy3, im_height - 1.0f), 0.0f);
    out_pred_boxes[index * 9 + 6] = max(min(x4 + dx4, im_width - 1.0f), 0.0f);
    out_pred_boxes[index * 9 + 7] = max(min(y4 + dy4, im_height - 1.0f), 0.0f);

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 9 + 8] = -1.0f;
    }
  }
}

__device__ inline bool isPolygonConvex(const float * box) {
  const float PI = 3.141593;
  const float TWO_PI = 2 * PI;
  // points is 'strictly convex': points are valid, side lengths non-zero, interior angles are strictly between zero and a straight
  // angle, and the polygon does not intersect itself.
  // NOTES:  1.  Algorithm: the signed changes of the direction angles from one side to the next side must be all positive or
  // all negative, and their sum must equal plus-or-minus one full turn (2 pi radians). Also check for too few,
  // invalid, or repeated points.
  //      2.  No check is explicitly done for zero internal angles(180 degree direction-change angle) as this is covered
  // in other ways, including the `n < 3` check.

  // Get starting information
  float old_x = box[4], old_y = box[5];
  float new_x = box[6], new_y = box[7];
  float new_direction = atan2(new_y - old_y, new_x - old_x);
  float old_direction;
  float angle_sum = 0.0, orientation=0;
  // Check each point (the side ending there, its angle) and accum. angles for ndx, newpoint in enumerate(polygon):
  for (int i = 0; i < 4; i++)
  {
     // Update point coordinates and side directions, check side length
     old_x = new_x; old_y = new_y; old_direction = new_direction;
     new_x = box[i * 2]; new_y = box[i * 2 + 1];
     new_direction = atan2(new_y - old_y, new_x - old_x);
     if (old_x == new_x && old_y == new_y)
        return false; // repeated consecutive points
     // Calculate & check the normalized direction-change angle
     float angle = new_direction - old_direction;
     if (angle <= -PI)
        angle += TWO_PI;  // make it in half-open interval (-Pi, Pi]
     else if (angle > PI)
        angle -= TWO_PI;
     if (i == 0)  // if first time through loop, initialize orientation
     {
        if (angle == 0.0) return false;
        orientation = angle > 0 ? 1 : -1;
     }
     else  // if other time through loop, check orientation is stable
     if (orientation * angle <= 0)  // not both pos. or both neg.
        return false;
     // Accumulate the direction-change angle
     angle_sum += angle;
     // Check that the total number of full turns is plus-or-minus 1
  }
  return fabs(round(angle_sum / TWO_PI)) == 1;
}

// filter box with stride less than rpn_min_size
// also those non-convex boxes
// filter: set score to zero
// dets (n, 5)
template<typename Dtype>
__global__ void FilterBoxKernel(const int count,
                                float min_size,
                                Dtype* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    float x1 = dets[index * 9 + 0];
    float y1 = dets[index * 9 + 1];
    float x2 = dets[index * 9 + 2];
    float y2 = dets[index * 9 + 3];
    float x3 = dets[index * 9 + 4];
    float y3 = dets[index * 9 + 5];
    float x4 = dets[index * 9 + 6];
    float y4 = dets[index * 9 + 7];
    float xmin = fmin(fmin(x1, x2), fmin(x3, x4));
    float ymin = fmin(fmin(y1, y2), fmin(y3, y4));
    float xmax = fmax(fmax(x1, x2), fmax(x3, x4));
    float ymax = fmax(fmax(y1, y2), fmax(y3, y4));
    float iw = xmax - xmin + 1.0f;
    float ih = ymax - ymin + 1.0f;

    min_size = fmax(min_size, 4.0f); // force
    if (iw < min_size || ih < min_size) {
      dets[index * 9 + 8] = -1.0f;
    } else {
      float box[] = {x1, y1, x2, y2, x3, y3, x4, y4};
      if ( ! isPolygonConvex(box)) {
        dets[index * 9 + 8] = -1.0f;
      }
    }
  }
}

// copy score and init order
// dets (n, 5); score (n, ); order (n, )
// count should be n (total anchors or proposals)
template<typename Dtype>
__global__ void CopyScoreKernel(const int count,
                                const Dtype* dets,
                                Dtype* score,
                                int* order) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    score[index] = dets[index * 9 + 8];
    order[index] = index;
  }
}

// reorder proposals according to order and keep the top_n proposals
// prev_dets (n, 5); order (n, ); dets (n, 5)
// count should be output anchor numbers (top_n)
template<typename Dtype>
__global__ void ReorderProposalsKernel(const int count,
                                       const Dtype* prev_dets,
                                       const int* order,
                                       Dtype* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    const int order_i = order[index];
    for (int j = 0; j < 9; j ++) {
      dets[index * 9 + j] = prev_dets[order_i * 9 + j];
    }
  }
}

// polygon nms
__device__ inline float trangle_area(float const * a, float const * b, float const * c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))/2.0;
}
__device__ inline float area(float const * int_pts, int num_of_inter) {
  float area = 0.0;
  for(int i = 0;i < num_of_inter - 2;i++) {
    area += fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}
__device__ inline void reorder_pts(float * int_pts, int num_of_inter) {
  if(num_of_inter > 0) {

    float center[2];

    center[0] = 0.0;
    center[1] = 0.0;

    for(int i = 0;i < num_of_inter;i++) {
      center[0] += int_pts[2 * i];
      center[1] += int_pts[2 * i + 1];
    }
    center[0] /= num_of_inter;
    center[1] /= num_of_inter;

    float vs[16];
    float v[2];
    float d;
    for(int i = 0;i < num_of_inter;i++) {
      v[0] = int_pts[2 * i]-center[0];
      v[1] = int_pts[2 * i + 1]-center[1];
      d = sqrt(v[0] * v[0] + v[1] * v[1]);
      v[0] = v[0] / d;
      v[1] = v[1] / d;
      if(v[1] < 0) {
        v[0]= - 2 - v[0];
      }
      vs[i] = v[0];
    }

    float temp,tx,ty;
    int j;
    for(int i=1;i<num_of_inter;++i){
      if(vs[i-1]>vs[i]){
        temp = vs[i];
        tx = int_pts[2*i];
        ty = int_pts[2*i+1];
        j=i;
        while(j>0&&vs[j-1]>temp){
          vs[j] = vs[j-1];
          int_pts[j*2] = int_pts[j*2-2];
          int_pts[j*2+1] = int_pts[j*2-1];
          j--;
        }
        vs[j] = temp;
        int_pts[j*2] = tx;
        int_pts[j*2+1] = ty;
      }
    }
  }

}
__device__ inline bool inter2line(float const * pts1, float const *pts2, int i, int j, float * temp_pts) {

  float a[2];
  float b[2];
  float c[2];
  float d[2];

  float area_abc, area_abd, area_cda, area_cdb;

  a[0] = pts1[2 * i];
  a[1] = pts1[2 * i + 1];

  b[0] = pts1[2 * ((i + 1) % 4)];
  b[1] = pts1[2 * ((i + 1) % 4) + 1];

  c[0] = pts2[2 * j];
  c[1] = pts2[2 * j + 1];

  d[0] = pts2[2 * ((j + 1) % 4)];
  d[1] = pts2[2 * ((j + 1) % 4) + 1];

  area_abc = trangle_area(a, b, c);
  area_abd = trangle_area(a, b, d);

  if(area_abc * area_abd >= -1e-5) {
    return false;
  }

  area_cda = trangle_area(c, d, a);
  area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= -1e-5) {
    return false;
  }
  float t = area_cda / (area_abd - area_abc);

  float dx = t * (b[0] - a[0]);
  float dy = t * (b[1] - a[1]);
  temp_pts[0] = a[0] + dx;
  temp_pts[1] = a[1] + dy;

  return true;
}

__device__ inline bool in_rect(float pt_x, float pt_y, float const * pts) {
    // https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
    float minX = fmin(fmin(pts[0],pts[2]),fmin(pts[4],pts[6]));
    float maxX = fmax(fmax(pts[0],pts[2]),fmax(pts[4],pts[6]));
    float minY = fmin(fmin(pts[1],pts[3]),fmin(pts[5],pts[7]));
    float maxY = fmax(fmax(pts[1],pts[3]),fmax(pts[5],pts[7]));
    if (pt_x < minX || pt_x > maxX || pt_y < minY || pt_y > maxY) {
        return false;
    }
    bool isInside = false;
    // the previous bounding box check can remove false-negatives in edge-cases
    // remove the previous check code to speed up if you don't care of edge-cases
    int n = 4; // point num
    for (int i = 0, j = n - 1; i < n; j = i++) {
        float ix = pts[i * 2], iy = pts[i * 2 + 1];
        float jx = pts[j * 2], jy = pts[j * 2 + 1];
        if ( (iy > pt_y) != (jy > pt_y) &&
                pt_x < (jx - ix) * (pt_y - iy) / (jy - iy) + ix ) {
            isInside = !isInside;
        }
    }

    return isInside;
}

__device__ inline int inter_pts(float const * pts1, float const * pts2, float * int_pts) {

  int num_of_inter = 0;

  for(int i = 0;i < 4;i++) {
    if(in_rect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
     if(in_rect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }
  }

  float temp_pts[2];

  for(int i = 0;i < 4;i++) {
    for(int j = 0;j < 4;j++) {
      bool has_pts = inter2line(pts1, pts2, i, j, temp_pts);
      if(has_pts) {
        int_pts[num_of_inter * 2] = temp_pts[0];
        int_pts[num_of_inter * 2 + 1] = temp_pts[1];
        num_of_inter++;
      }
    }
  }


  return num_of_inter;
}

__device__ inline float devRotateIoU(float const * const region1, float const * const region2) {
  // enlarge to decrease the edge cases
  const float pts1[] = {region1[0] * 100, region1[1] * 100, region1[2] * 100, region1[3] * 100,
                         region1[4] * 100, region1[5] * 100, region1[6] * 100, region1[7] * 100};
  const float pts2[] = {region2[0] * 100, region2[1] * 100, region2[2] * 100, region2[3] * 100,
                         region2[4] * 100, region2[5] * 100, region2[6] * 100, region2[7] * 100};

  float area1 = area(pts1, 4);
  float area2 = area(pts2, 4);
  //float area_inter = inter(pts1, pts2);
  float int_pts[16];
  int num_of_inter;

  num_of_inter = inter_pts(pts1, pts2, int_pts);

  reorder_pts(int_pts, num_of_inter);

  float area_inter = area(int_pts, num_of_inter);

  float result = area_inter / (area1 + area2 - area_inter + 1e-5);

//  if(result < 0) {
//    result = 0.0;
//  }
  return result;
}

__global__ void rotate_nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, uint64_t *dev_mask) {
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 9];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 9 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 0];
    block_boxes[threadIdx.x * 9 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 1];
    block_boxes[threadIdx.x * 9 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 2];
    block_boxes[threadIdx.x * 9 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 3];
    block_boxes[threadIdx.x * 9 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 4];
    block_boxes[threadIdx.x * 9 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 5];
    block_boxes[threadIdx.x * 9 + 6] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 6];
    block_boxes[threadIdx.x * 9 + 7] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 7];
    block_boxes[threadIdx.x * 9 + 8] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 8];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 9;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devRotateIoU(cur_box, block_boxes + i * 9) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _nms(const mshadow::Tensor<gpu, 2>& boxes,
          const float nms_overlap_thresh,
          int *keep,
          int *num_out) {
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int boxes_num = boxes.size(0);
  const int boxes_dim = boxes.size(1);

  float* boxes_dev = boxes.dptr_;
  uint64_t* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  FRCNN_CUDA_CHECK(cudaMalloc(&mask_dev,
                              boxes_num * col_blocks * sizeof(uint64_t)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  rotate_nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);
  FRCNN_CUDA_CHECK(cudaPeekAtLastError());
  std::vector<uint64_t> mask_host(boxes_num * col_blocks);
  FRCNN_CUDA_CHECK(cudaMemcpy(&mask_host[0],
                              mask_dev,
                              sizeof(uint64_t) * boxes_num * col_blocks,
                              cudaMemcpyDeviceToHost));

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep[num_to_keep++] = i;
      uint64_t *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  FRCNN_CUDA_CHECK(cudaFree(mask_dev));
}

// copy proposals to output
// dets (top_n, 5); keep (top_n, ); out (top_n, )
// count should be top_n (total anchors or proposals)
template<typename Dtype>
__global__ void PrepareOutput(const int count,
                              const Dtype* dets,
                              const int* keep,
                              const int out_size,
                              Dtype* out,
                              Dtype* out_h,
                              Dtype* score) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    out[index * 9] = 0; // batch inds are 0
    out_h[index * 5] = 0;
    int keep_i;
    if (index < out_size) {
      keep_i = keep[index];
    } else {
      keep_i = keep[index % out_size];
    }
    int s = keep_i * 9;
    for (int j = 0; j < 8; ++j) {
      out[index * 9 + j + 1] = dets[s + j];
    }
    Dtype xmin = fmin(fmin(dets[s + 0], dets[s + 2]), fmin(dets[s + 4], dets[s + 6]));
    Dtype xmax = fmax(fmax(dets[s + 0], dets[s + 2]), fmax(dets[s + 4], dets[s + 6]));
    Dtype ymin = fmin(fmin(dets[s + 1], dets[s + 3]), fmin(dets[s + 5], dets[s + 7]));
    Dtype ymax = fmax(fmax(dets[s + 1], dets[s + 3]), fmax(dets[s + 5], dets[s + 7]));
    out_h[index * 5 + 1] = xmin;
    out_h[index * 5 + 2] = ymin;
    out_h[index * 5 + 3] = xmax;
    out_h[index * 5 + 4] = ymax;
    score[index] = dets[keep_i * 9 + 8];
  }
}

}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu>
class ProposalRotateGPUOp : public Operator{
 public:
  explicit ProposalRotateGPUOp(ProposalRotateParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[proposal::kOut], kWriteTo);
    CHECK_EQ(in_data[proposal::kClsProb].shape_[0], 1)
      << "Sorry, multiple images each device is not implemented.";

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Shape<4> fg_scores_shape = Shape4(in_data[proposal::kClsProb].shape_[0],
                                      in_data[proposal::kClsProb].shape_[1] / 2,
                                      in_data[proposal::kClsProb].shape_[2],
                                      in_data[proposal::kClsProb].shape_[3]);
    real_t* foreground_score_ptr = in_data[proposal::kClsProb].dptr<real_t>()
                                    + fg_scores_shape.Size();
    Tensor<xpu, 4> scores = Tensor<xpu, 4>(foreground_score_ptr, fg_scores_shape);
    Tensor<xpu, 4> bbox_deltas = in_data[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> im_info = in_data[proposal::kImInfo].get<xpu, 2, real_t>(s);

    Tensor<xpu, 2> out = out_data[proposal::kOut].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out_h = out_data[proposal::kOutH].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out_score = out_data[proposal::kScore].get<xpu, 2, real_t>(s);

    int num_anchors = in_data[proposal::kClsProb].shape_[1] / 2;
    int height = scores.size(2);
    int width = scores.size(3);
    int count = num_anchors * height * width;  // count of total anchors
    // set to -1 for max
    int rpn_pre_nms_top_n = (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count);
    int rpn_post_nms_top_n = std::min(param_.rpn_post_nms_top_n, rpn_pre_nms_top_n);

    // Generate first anchors based on base anchor
    std::vector<float> base_anchor(4);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = param_.feature_stride - 1.0;
    base_anchor[3] = param_.feature_stride - 1.0;
    CHECK_EQ(num_anchors, param_.ratios.info.size() * param_.scales.info.size() * param_.angles.info.size());
    std::vector<float> anchors;
    utils::GenerateAnchors(base_anchor,
                           param_.ratios.info,
                           param_.scales.info,
                           param_.angles.info,
                           &anchors);

    // Copy generated anchors to GPU
    float* workspace_proposals_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&workspace_proposals_ptr, sizeof(float) * count * 9));
    Tensor<xpu, 2> workspace_proposals(workspace_proposals_ptr, Shape2(count, 9));
    FRCNN_CUDA_CHECK(cudaMemcpy(workspace_proposals.dptr_,
                                &anchors[0], sizeof(float) * anchors.size(),
      cudaMemcpyHostToDevice));

    // Copy proposals to a mesh grid
    dim3 dimGrid((count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
    dim3 dimBlock(kMaxThreadsPerBlock);
    CheckLaunchParam(dimGrid, dimBlock, "ProposalGrid");
    ProposalGridKernel<<<dimGrid, dimBlock>>>(
      count, num_anchors, height, width, param_.feature_stride,
      scores.dptr_, workspace_proposals.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // im_info is small, we want to copy them to cpu
    std::vector<float> cpu_im_info(3);
    FRCNN_CUDA_CHECK(cudaMemcpy(&cpu_im_info[0], im_info.dptr_,
                                sizeof(float) * cpu_im_info.size(),
                                cudaMemcpyDeviceToHost));

    // prevent padded predictions
    int real_height = static_cast<int>(cpu_im_info[0] / param_.feature_stride);
    int real_width = static_cast<int>(cpu_im_info[1] / param_.feature_stride);
    CHECK_GE(height, real_height) << height << " " << real_height << std::endl;
    CHECK_GE(width, real_width) << width << " " << real_width << std::endl;

    // Transform anchors and bbox_deltas into bboxes
    CheckLaunchParam(dimGrid, dimBlock, "BBoxPred");
    if (param_.iou_loss) {
      IoUPredKernel<<<dimGrid, dimBlock>>>(
        count, num_anchors, height, width, real_height, real_width,
        cpu_im_info[0], cpu_im_info[1],
        workspace_proposals.dptr_, bbox_deltas.dptr_, workspace_proposals.dptr_);
    } else {
      BBoxPredKernel<<<dimGrid, dimBlock>>>(
        count, num_anchors, height, width, real_height, real_width,
        cpu_im_info[0], cpu_im_info[1],
        workspace_proposals.dptr_, bbox_deltas.dptr_, workspace_proposals.dptr_);
    }
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // filter boxes with less than rpn_min_size
    CheckLaunchParam(dimGrid, dimBlock, "FilterBox");
    FilterBoxKernel<<<dimGrid, dimBlock>>>(
      count, param_.rpn_min_size * cpu_im_info[2], workspace_proposals.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // Copy score to a continuous memory
    float* score_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&score_ptr, sizeof(float) * count));
    Tensor<xpu, 1> score(score_ptr, Shape1(count));
    int* order_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&order_ptr, sizeof(int) * count));
    Tensor<xpu, 1, int> order(order_ptr, Shape1(count));

    CheckLaunchParam(dimGrid, dimBlock, "CopyScore");
    CopyScoreKernel<<<dimGrid, dimBlock>>>(
      count, workspace_proposals.dptr_, score.dptr_, order.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // argsort score, save order
    thrust::stable_sort_by_key(thrust::device,
                               score.dptr_,
                               score.dptr_ + score.size(0),
                               order.dptr_,
                               thrust::greater<real_t>());
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // Reorder proposals according to order
    float* workspace_ordered_proposals_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&workspace_ordered_proposals_ptr,
                                sizeof(float) * rpn_pre_nms_top_n * 9));
    Tensor<xpu, 2> workspace_ordered_proposals(workspace_ordered_proposals_ptr,
                                               Shape2(rpn_pre_nms_top_n, 9));

    dimGrid.x = (rpn_pre_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    CheckLaunchParam(dimGrid, dimBlock, "ReorderProposals");
    ReorderProposalsKernel<<<dimGrid, dimBlock>>>(
      rpn_pre_nms_top_n, workspace_proposals.dptr_, order.dptr_, workspace_ordered_proposals.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    FRCNN_CUDA_CHECK(cudaFree(workspace_proposals_ptr));
    FRCNN_CUDA_CHECK(cudaFree(score_ptr));
    FRCNN_CUDA_CHECK(cudaFree(order_ptr));

    // perform nms
    std::vector<int> _keep(workspace_ordered_proposals.size(0));
    int out_size = 0;
    _nms(workspace_ordered_proposals,
         param_.threshold,
         &_keep[0],
         &out_size);

    // copy nms result to gpu
    int* keep;
    FRCNN_CUDA_CHECK(cudaMalloc(&keep, sizeof(int) * _keep.size()));
    FRCNN_CUDA_CHECK(cudaMemcpy(keep, &_keep[0], sizeof(int) * _keep.size(),
                                cudaMemcpyHostToDevice));

    // copy results after nms
    dimGrid.x = (rpn_post_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    CheckLaunchParam(dimGrid, dimBlock, "PrepareOutput");
    PrepareOutput<<<dimGrid, dimBlock>>>(
      rpn_post_nms_top_n, workspace_ordered_proposals.dptr_, keep, out_size,
      out.dptr_, out_h.dptr_, out_score.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // free temporary memory
    FRCNN_CUDA_CHECK(cudaFree(keep));
    FRCNN_CUDA_CHECK(cudaFree(workspace_ordered_proposals_ptr));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 3);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[proposal::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[proposal::kImInfo].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[proposal::kClsProb], 0);
    Assign(gbbox, req[proposal::kBBoxPred], 0);
    Assign(ginfo, req[proposal::kImInfo], 0);
  }

 private:
  ProposalRotateParam param_;
};  // class ProposalGPUOp

template<>
Operator* CreateOp<gpu>(ProposalRotateParam param) {
  return new ProposalRotateGPUOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet

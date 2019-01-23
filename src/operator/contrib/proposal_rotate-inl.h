/*!
 * Copyright (c) 2015 by Contributors
 * \file proposal-inl.h
 * \brief Proposal Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo
*/
#ifndef MXNET_OPERATOR_CONTRIB_PROPOSAL_ROTATE_INL_H_
#define MXNET_OPERATOR_CONTRIB_PROPOSAL_ROTATE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <cmath>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"

// extend NumericalParam
namespace mxnet {
namespace op {

/*!
* \brief structure for numerical tuple input
* \tparam VType data type of param
*/
template<typename VType>
struct NumericalParam {
  NumericalParam() {}
  explicit NumericalParam(VType *begin, VType *end) {
    int32_t size = static_cast<int32_t>(end - begin);
    info.resize(size);
    for (int i = 0; i < size; ++i) {
      info[i] = *(begin + i);
    }
  }
  inline size_t ndim() const {
    return info.size();
  }
  std::vector<VType> info;
};

template<typename VType>
inline std::istream &operator>>(std::istream &is, NumericalParam<VType> &param) {
  while (true) {
    char ch = is.get();
    if (ch == '(') break;
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  VType idx;
  std::vector<VType> tmp;
  // deal with empty case
  size_t pos = is.tellg();
  char ch = is.get();
  if (ch == ')') {
    param.info = tmp;
    return is;
  }
  is.seekg(pos);
  // finish deal
  while (is >> idx) {
    tmp.push_back(idx);
    char ch;
    do {
      ch = is.get();
    } while (isspace(ch));
    if (ch == ',') {
      while (true) {
        ch = is.peek();
        if (isspace(ch)) {
          is.get(); continue;
        }
        if (ch == ')') {
          is.get(); break;
        }
        break;
      }
      if (ch == ')') break;
    } else if (ch == ')') {
      break;
    } else {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  param.info = tmp;
  return is;
}

template<typename VType>
inline std::ostream &operator<<(std::ostream &os, const NumericalParam<VType> &param) {
  os << '(';
  for (index_t i = 0; i < param.info.size(); ++i) {
    if (i != 0) os << ',';
    os << param.info[i];
  }
  // python style tuple
  if (param.info.size() == 1) os << ',';
  os << ')';
  return os;
}

}  // namespace op
}  // namespace mxnet

namespace mxnet {
namespace op {

namespace proposal {
enum ProposalOpInputs {kClsProb, kBBoxPred, kImInfo};
enum ProposalOpOutputs {kOut, kOutH, kScore};
enum ProposalForwardResource {kTempResource};
}  // proposal

struct ProposalRotateParam : public dmlc::Parameter<ProposalRotateParam> {
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float threshold;
  int rpn_min_size;
  NumericalParam<float> scales;
  NumericalParam<float> ratios;
  NumericalParam<float> angles;
  int feature_stride;
  bool output_score;
  bool output_horizon_rois;
  bool iou_loss;
  DMLC_DECLARE_PARAMETER(ProposalRotateParam) {
    float tmp[] = {0, 0, 0, 0};
    DMLC_DECLARE_FIELD(rpn_pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(rpn_post_nms_top_n).set_default(300)
    .describe("Overlap threshold used for non-maximum"
              "suppresion(suppress boxes with IoU >= this threshold");
    DMLC_DECLARE_FIELD(threshold).set_default(0.7)
    .describe("NMS value, below which to suppress.");
    DMLC_DECLARE_FIELD(rpn_min_size).set_default(16)
    .describe("Minimum height or width in proposal");
    tmp[0] = 4.0f; tmp[1] = 8.0f; tmp[2] = 16.0f; tmp[3] = 32.0f;
    DMLC_DECLARE_FIELD(scales).set_default(NumericalParam<float>(tmp, tmp + 4))
    .describe("Used to generate anchor windows by enumerating scales");
    tmp[0] = 0.5f; tmp[1] = 1.0f; tmp[2] = 2.0f;
    DMLC_DECLARE_FIELD(ratios).set_default(NumericalParam<float>(tmp, tmp + 3))
    .describe("Used to generate anchor windows by enumerating ratios");
    tmp[0] = -60.f; tmp[1] = -30.f; tmp[2] = 0.f;
    DMLC_DECLARE_FIELD(angles).set_default(NumericalParam<float>(tmp, tmp + 3))
    .describe("Used to generate anchor windows by enumerating angles");
    DMLC_DECLARE_FIELD(feature_stride).set_default(16)
    .describe("The size of the receptive field each unit in the convolution layer of the rpn,"
              "for example the product of all stride's prior to this layer.");
    DMLC_DECLARE_FIELD(output_score).set_default(false)
    .describe("Add score to outputs");
    DMLC_DECLARE_FIELD(output_horizon_rois).set_default(false)
    .describe("Add horizon_rois to outputs");
    DMLC_DECLARE_FIELD(iou_loss).set_default(false)
    .describe("Usage of IoU Loss");
  }
};

template<typename xpu>
Operator *CreateOp(ProposalRotateParam param);

#if DMLC_USE_CXX11
class ProposalRotateProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[cls_prob, bbox_pred, im_info]";
    const TShape &dshape = in_shape->at(proposal::kClsProb);
    if (dshape.ndim() == 0) return false;
    Shape<4> bbox_pred_shape; // quandrangle 8 num
    bbox_pred_shape = Shape4(dshape[0], dshape[1] / 2 * 8, dshape[2], dshape[3]);
    SHAPE_ASSIGN_CHECK(*in_shape, proposal::kBBoxPred,
                       bbox_pred_shape);
    Shape<2> im_info_shape;
    im_info_shape = Shape2(dshape[0], 3);
    SHAPE_ASSIGN_CHECK(*in_shape, proposal::kImInfo, im_info_shape);
    out_shape->clear();
    // output
    // box_dim = 8
    out_shape->push_back(Shape2(param_.rpn_post_nms_top_n, 8 + 1));
    out_shape->push_back(Shape2(param_.rpn_post_nms_top_n, 4 + 1));
    // score
    out_shape->push_back(Shape2(param_.rpn_post_nms_top_n, 1));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ProposalRotateProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_ProposalRotate";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  int NumVisibleOutputs() const override {
    int n = 1;
    if (param_.output_score) ++n;
    if (param_.output_horizon_rois) ++n;
    return n;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"cls_prob", "bbox_pred", "im_info"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "output_horizon", "score"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ProposalRotateParam param_;
};  // class ProposalRotateProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

//========================
// Anchor Generation Utils
//========================
namespace mxnet {
namespace op {
namespace utils {

inline float get_distance(float px1, float py1, float px2, float py2, float px3, float py3, float px4, float py4,
                          float qx1, float qy1, float qx2, float qy2, float qx3, float qy3, float qx4, float qy4) {
    return (px1-qx1)*(px1-qx1) + (py1-qy1)*(py1-qy1) + (px2-qx2)*(px2-qx2) + (py2-qy2)*(py2-qy2)
         + (px3-qx3)*(px3-qx3) + (py3-qy3)*(py3-qy3) + (px4-qx4)*(px4-qx4) + (py4-qy4)*(py4-qy4);
}

inline void _MakeAnchor(float w,
                        float h,
                        float x_ctr,
                        float y_ctr,
                        float angle,
                        std::vector<float> *out_anchors) {
  // code from OpenCV: void RotatedRect::points(Point2f pt[]) const
  // RotatedRect degree range -90~0, so if degree > 0, swap width with height
  if (angle > 0) std::swap(w, h);
  float _angle = angle * 3.1415926535 / 180.0; // rad

  float b = cos(_angle) * 0.5;
  float a = sin(_angle) * 0.5;

  float pt0x = x_ctr - a * h - b * w;
  float pt0y = y_ctr + b * h - a * w;
  float pt1x = x_ctr + a * h - b * w;
  float pt1y = y_ctr - b * h - a * w;
  float pt2x = 2 * x_ctr - pt0x;
  float pt2y = 2 * y_ctr - pt0y;
  float pt3x = 2 * x_ctr - pt1x;
  float pt3y = 2 * y_ctr - pt1y;
  float anchor[] = {pt0x, pt0y, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y};
  // get_best_begin_point
  float xmin = std::min(std::min(pt0x, pt1x), std::min(pt2x, pt3x));
  float ymin = std::min(std::min(pt0y, pt1y), std::min(pt2y, pt3y));
  float xmax = std::max(std::max(pt0x, pt1x), std::max(pt2x, pt3x));
  float ymax = std::max(std::max(pt0y, pt1y), std::max(pt2y, pt3y));
  int idx = 0;
  float min_dis = FLT_MAX;
  for (int i=0;i<4;i++) {
    float dis = get_distance(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax,
                 anchor[(i*2+0)%8], anchor[(i*2+1)%8], anchor[(i*2+2)%8], anchor[(i*2+3)%8],
                 anchor[(i*2+4)%8], anchor[(i*2+5)%8], anchor[(i*2+6)%8], anchor[(i*2+7)%8]);
    if (dis < min_dis) {
      min_dis = dis;
      idx = i;
    }
  }

  out_anchors->push_back(anchor[(idx*2+0)%8]);
  out_anchors->push_back(anchor[(idx*2+1)%8]);
  out_anchors->push_back(anchor[(idx*2+2)%8]);
  out_anchors->push_back(anchor[(idx*2+3)%8]);
  out_anchors->push_back(anchor[(idx*2+4)%8]);
  out_anchors->push_back(anchor[(idx*2+5)%8]);
  out_anchors->push_back(anchor[(idx*2+6)%8]);
  out_anchors->push_back(anchor[(idx*2+7)%8]);
  out_anchors->push_back(0.0f);
}

inline void _Transform(float scale,
                       float ratio,
                       float angle,
                       const std::vector<float>& base_anchor,
                       std::vector<float>  *out_anchors) {
  float w = base_anchor[2] - base_anchor[0] + 1.0f;
  float h = base_anchor[3] - base_anchor[1] + 1.0f;
  float x_ctr = base_anchor[0] + 0.5 * (w - 1.0f);
  float y_ctr = base_anchor[1] + 0.5 * (h - 1.0f);
  float size = w * h;
  float size_ratios = std::floor(size / ratio);
  float new_w = std::floor(std::sqrt(size_ratios) + 0.5f) * scale;
  float new_h = std::floor((new_w / scale * ratio) + 0.5f) * scale;

  _MakeAnchor(new_w, new_h, x_ctr, y_ctr, angle, out_anchors);
}

// out_anchors must have shape (n, 9), where n is ratios.size() * scales.size() * angles.size()
inline void GenerateAnchors(const std::vector<float>& base_anchor,
                            const std::vector<float>& ratios,
                            const std::vector<float>& scales,
                            const std::vector<float>& angles,
                            std::vector<float> *out_anchors) {
  for (size_t j = 0; j < ratios.size(); ++j) {
    for (size_t k = 0; k < scales.size(); ++k) {
      for (size_t t = 0; t < angles.size(); ++t) {
        _Transform(scales[k], ratios[j], angles[t], base_anchor, out_anchors);
      }
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_PROPOSAL_ROTATE_INL_H_

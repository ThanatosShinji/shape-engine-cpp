// x64_inference_engine.cpp: 目标的源文件。
//

#include "x64_inference_engine.h"
#include "jblas/jit_blas_gemm.h"
#include "jblas/jit_blas_wrapper.h"
#include "jblas_common.hpp"
#include "jblas_gemm.h"
#include <numeric>

using namespace x64_inference_engine_f32;

std::vector<LayerCreator *> LayerFactory::mCreatetors;

void RuntimeEngine::set_threads(int num) {
  ne_jblas::ne_threading::get()->set_threads(num);
}

size_t shape_volume(int *ptr, int n) {
  size_t vol = 1;
  for (size_t i = 0; i < n; i++) {
    vol *= ptr[i];
  }
  return vol;
}

size_t shape_volume(tensor_shape_t &shape) {
  return shape_volume(shape.ptr, shape.n);
}

class Conv : public LayerBase {
public:
  Conv(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {
    mDilations = attrs["dilations"].mInts;
    mGroup = attrs["group"].mInts;
    mKernel = attrs["kernel_shape"].mInts;
    mPads = attrs["pads"].mInts;
    mStrides = attrs["strides"].mInts;
    if (attrs["postop_count"].mInts.size() > 0) {
      mPostopCount = attrs["postop_count"].mInts[0];
      for (int i = 0; i < mPostopCount; i++) {
        std::string key = std::string("postop_") + std::to_string(i);
        mPostops.push_back(attrs[key].mStrings[0]);
      }
    } else {
      mPostopCount = 0;
    }
  }

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    int batch = xshape.ptr[0];
    int h = xshape.ptr[2];
    int w = xshape.ptr[3];
    int oh = yshape.ptr[2];
    int ow = yshape.ptr[3];

    for (int i = 0; i < batch; i++) {
      auto xbatchptr = xptr + i * batch * mIC * h * w;
      auto ybatchptr = yptr + i * batch * mOC * oh * ow;
      for (int j = 0; j < mOC; j++) {
        float bias = 0.f;
        if (mBias.size()) {
          bias = mBias[j];
        }
        for (int k = 0; k < oh; k++) {
          auto yhptr = ybatchptr + j * oh * ow + k * ow;
          for (int l = 0; l < ow; l++) {
            float tmp = bias;
            for (int iy = 0; iy < mKernel[0]; iy++) {
              auto srcy = k * mStrides[0] + iy - mPads[0];
              if (srcy < 0 || srcy >= h) {
                continue;
              }
              auto xhptr = xbatchptr + srcy * w;
              for (int ix = 0; ix < mKernel[1]; ix++) {
                auto srcx = l * mStrides[1] + ix - mPads[1];
                if (srcx >= 0 && srcx < w) {
                  for (int icn = 0; icn < mIC; icn++) {
                    auto xval = xhptr[srcx + icn * h * w];
                    auto wval =
                        mWeights[j * mKernel[0] * mKernel[1] * mIC +
                                 iy * mKernel[1] * mIC + ix * mIC + icn];
                    tmp += xval * wval;
                  }
                }
              }
            }
            for (int ipost = 0; ipost < mPostopCount; ipost++) {
              if (std::strcmp(mPostops[ipost].c_str(), "Relu") == 0) {
                tmp = std::max(0.f, tmp);
              }
              if (std::strcmp(mPostops[ipost].c_str(), "Add") == 0) {
                auto post1ptr = (float *)mInputs[1];
                tmp += post1ptr[i * batch * mOC * oh * ow + j * oh * ow +
                                k * ow + l];
              }
            }
            yhptr[l] = tmp;
          }
        }
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    auto &weight = tensors[0];
    mIC = weight.mShape[1];
    mOC = weight.mShape[0];
    auto newsize = mIC * mOC * mKernel[0] * mKernel[1];
    mWeights.resize(newsize);
    auto rawptr = (float *)weight.mRawptr;
    // tranpose from Oc*Ic*Ky*Kx to Oc*Ky*Kx*Ic
    for (int io = 0; io < mOC; io++) {
      for (int ii = 0; ii < mIC; ii++) {
        for (int iky = 0; iky < mKernel[0]; iky++) {
          for (int ikx = 0; ikx < mKernel[1]; ikx++) {
            mWeights[io * mKernel[0] * mKernel[1] * mIC +
                     iky * mKernel[1] * mIC + ikx * mIC + ii] =
                *(rawptr + io * mKernel[0] * mKernel[1] * mIC +
                  ii * mKernel[0] * mKernel[1] + iky * mKernel[1] + ikx);
          }
        }
      }
    }
    if (tensors.size() > 1) {
      auto &bias = tensors[1];
      mBias.resize(mOC);
      memcpy(mBias.data(), bias.mRawptr, bias.mSize);
    }
  }

  int mIC, mOC, mPostopCount;
  std::vector<int> mDilations, mGroup, mKernel, mPads, mStrides;
  std::vector<float> mWeights, mBias;
  std::vector<std::string> mPostops;
};
REGISTER_LAYER(Conv);

class BatchNormalization : public LayerBase {
public:
  BatchNormalization(const char *_name, attr_map_t &attrs)
      : LayerBase(_name, attrs) {
    mSpatial = attrs["spatial"].mInts;
    mEpsilon = attrs["epsilon"].mFloats;
    mMomentum = attrs["momentum"].mFloats;
    if (attrs["postop_count"].mInts.size() > 0) {
      for (int i = 0; i < attrs["postop_count"].mInts[0]; i++) {
        std::string key = std::string("postop_") + std::to_string(i);
        mPostops.push_back(attrs[key].mStrings[0]);
      }
    }
  }

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    int batch = xshape.ptr[0];
    int cn = xshape.ptr[1];
    int h = xshape.ptr[2];
    int w = xshape.ptr[3];
    for (int i = 0; i < batch; i++) {
      for (int j = 0; j < cn; j++) {
        auto sqrt_variance = std::sqrt(varptr[j] + mEpsilon[0]);
        auto sm = gammaptr[j] / sqrt_variance;
        auto sv = betaptr[j];
        auto mean = meanptr[j];
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            auto offset = i * cn * h * w + j * h * w + k * w + l;
            yptr[offset] = sm * (xptr[offset] - mean) + sv;
            for (int ipost = 0; ipost < mPostops.size(); ipost++) {
              if (std::strcmp(mPostops[ipost].c_str(), "Relu") == 0) {
                yptr[offset] = std::max(0.f, yptr[offset]);
              }
            }
          }
        }
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    auto &gamma = tensors[0];
    auto &beta = tensors[1];
    auto &mean = tensors[2];
    auto &var = tensors[3];
    mOC = gamma.mShape[0];
    mWeights.resize(mOC * 4);
    gammaptr = mWeights.data();
    betaptr = gammaptr + mOC;
    meanptr = betaptr + mOC;
    varptr = meanptr + mOC;
    memcpy(gammaptr, gamma.mRawptr, mOC * sizeof(float));
    memcpy(betaptr, beta.mRawptr, mOC * sizeof(float));
    memcpy(meanptr, mean.mRawptr, mOC * sizeof(float));
    memcpy(varptr, var.mRawptr, mOC * sizeof(float));
  }

  int mOC;
  float *gammaptr, *betaptr, *meanptr, *varptr;
  std::vector<float> mWeights;
  std::vector<int> mSpatial;
  std::vector<float> mEpsilon, mMomentum;
  std::vector<std::string> mPostops;
};
REGISTER_LAYER(BatchNormalization);

class Relu : public LayerBase {
public:
  Relu(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    for (int i = 0; i < xshape.ptr[0]; i++) {
      for (int j = 0; j < xshape.ptr[1]; j++) {
        for (int k = 0; k < xshape.ptr[2]; k++) {
          for (int l = 0; l < xshape.ptr[3]; l++) {
            auto offset = i * xshape.ptr[1] * xshape.ptr[2] * xshape.ptr[3] +
                          j * xshape.ptr[2] * xshape.ptr[3] +
                          k * xshape.ptr[3] + l;
            yptr[offset] = std::max(xptr[offset], 0.f);
          }
        }
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {}
};
REGISTER_LAYER(Relu);

class Flatten : public LayerBase {
public:
  Flatten(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    auto size = 1;
    for (int i = 0; i < xshape.n; i++) {
      size *= xshape.ptr[i];
    }
    std::memcpy(yptr, xptr, size * sizeof(float));
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {}
};
REGISTER_LAYER(Flatten);

class MaxPool : public LayerBase {
public:
  MaxPool(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {
    mKernel = attrs["kernel_shape"].mInts;
    mPads = attrs["pads"].mInts;
    mStrides = attrs["strides"].mInts;
  }

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    int batch = xshape.ptr[0];
    int cn = xshape.ptr[1];
    int h = xshape.ptr[2];
    int w = xshape.ptr[3];
    int oh = yshape.ptr[2];
    int ow = yshape.ptr[3];

    for (int i = 0; i < batch; i++) {
      auto xbatchptr = xptr + i * cn * h * w;
      auto ybatchptr = yptr + i * cn * oh * ow;
      for (int j = 0; j < cn; j++) {
        auto xcnptr = xbatchptr + j * h * w;
        auto ycnptr = ybatchptr + j * oh * ow;
        for (int k = 0; k < oh; k++) {
          auto yhptr = ycnptr + k * ow;
          for (int l = 0; l < ow; l++) {
            float tmp = std::numeric_limits<float>::min();
            for (int iy = 0; iy < mKernel[0]; iy++) {
              auto srcy = k * mStrides[0] + iy - mPads[0];
              if (srcy < 0 || srcy >= h) {
                continue;
              }
              auto xhptr = xcnptr + srcy * w;
              for (int ix = 0; ix < mKernel[1]; ix++) {
                auto srcx = l * mStrides[1] + ix - mPads[1];
                if (srcx >= 0 && srcx < w) {
                  auto srcval = xhptr[srcx];
                  tmp = std::max(tmp, srcval);
                }
              }
            }
            yhptr[l] = tmp;
          }
        }
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {}
  std::vector<int> mKernel, mPads, mStrides;
};
REGISTER_LAYER(MaxPool);

class GlobalAveragePool : public LayerBase {
public:
  GlobalAveragePool(const char *_name, attr_map_t &attrs)
      : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    int batch = xshape.ptr[0];
    int cn = xshape.ptr[1];
    int h = xshape.ptr[2];
    int w = xshape.ptr[3];
    for (int i = 0; i < batch; i++) {
      for (int j = 0; j < cn; j++) {
        auto tmp = 0.f;
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            tmp += xptr[i * cn * h * w + j * h * w + k * w + l];
          }
        }
        tmp /= h * w;
        yptr[i * cn + j] = tmp;
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {}
};
REGISTER_LAYER(GlobalAveragePool);

class Gemm : public LayerBase {
public:
  using tSGEMM_AVX2 = jblas::wrapper::gemm::LauncherBase<
      JblasAVX2, jblas::gemm::SCoreRowNAvx2<24, 4>,
      jblas::prologue_a::gemm::ActivationBase,
      jblas::prologue_b::gemm::WeightPack,
      jblas::epilogue::gemm::AlphaBetaProcessFp32>;
  Gemm(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {
    mAlpha = attrs["alpha"].mFloats;
    mBeta = attrs["beta"].mFloats;
    mTransA = attrs["transA"].mInts;
    if (mTransA.size() == 0) {
      mTransA.resize(1, 0);
    }
    mTransB = attrs["transB"].mInts;
    if (mTransB.size() == 0) {
      mTransB.resize(1, 0);
    }
  }

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    M = xshape.ptr[0];
    if (xshape.n > 2) {
      if (xshape.n == 4) {
        M = xshape.ptr[0] * xshape.ptr[1];
      } else {
        M = 1;
        for (size_t i = 0; i < xshape.n - 1; i++) {
          M *= xshape.ptr[i];
        }
      }
    }
    if (mTransA[0]) {
      M = xshape.ptr[1];
    }
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];

    if (mTransA[0] == 0) {
      jblas::utils::GemmProblem gp(1, M, N, K);
      tSGEMM_AVX2::Param args{gp,
                              {xptr, K},
                              {NULL, 0, &mPackedB},
                              {yptr, mBias.data(), N, 0, mAlpha[0], mBeta[0]}};
      using tParallel =
          jblas::parallel::gemm::SchedulerBase<tSGEMM_AVX2::GemmCore>;
      jblas::parallel::GemmRun<tParallel>(mKernelAVX2, args,
                                          ne_jblas::ne_threading::get());
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    auto &weight = tensors[0];
    if (mTransB[0]) {
      K = weight.mShape[1];
      N = weight.mShape[0];
    } else {
      K = weight.mShape[0];
      N = weight.mShape[1];
    }

    mPackedB = mKernelAVX2.mProB.createStorage(N, K);
    mPackedBuffer.resize(mPackedB.mSize);
    mPackedB.assign(mPackedBuffer.data());
    auto rawptr = (float *)weight.mRawptr;
    if (mTransB[0]) {
      mKernelAVX2.mProB.packWeightTranspose(N, K, {rawptr, K, &mPackedB},
                                            ne_jblas::ne_threading::get());
    } else {
      mKernelAVX2.mProB.packWeight(N, K, {rawptr, N, &mPackedB},
                                   ne_jblas::ne_threading::get());
    }
    mBias.resize(N);
    if (tensors.size() > 1) {
      auto &bias = tensors[1];
      memcpy(mBias.data(), bias.mRawptr, N * sizeof(float));
    } else {
      memset(mBias.data(), 0, mBias.size() * sizeof(mBias[0]));
    }
  }
  tSGEMM_AVX2 mKernelAVX2;
  tSGEMM_AVX2::PrologueB::StorageType mPackedB{0};
  int N, K, M;
  std::vector<float> mAlpha, mBeta;
  std::vector<int> mTransA, mTransB;
  std::vector<float> mBias;
  jblas::utils::avector<int8_t> mPackedBuffer;
};
REGISTER_LAYER(Gemm);

class Gather : public LayerBase {
public:
  Gather(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {
    mAxis = attrs["axis"].mInts;
    if (mAxis.size() == 0) {
      mAxis.resize(1, 0);
    }
  }

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (int *)mInputs[0];
    auto xshape = mIShapes[0];
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    if (mAxis[0] == 0 && mWeightShape.size() == 2) {
      for (size_t ib = 0; ib < yshape.ptr[0]; ib++) {
        for (int i = 0; i < yshape.ptr[1]; i++) {
          std::memcpy(
              yptr + ib * yshape.ptr[1] * yshape.ptr[2] + i * mWeightShape[1],
              mWeights.data() + xptr[ib * yshape.ptr[1] + i] * mWeightShape[1],
              mWeightShape[1] * sizeof(float));
        }
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    auto &weight = tensors[0];
    mWeightShape = weight.mShape;
    auto size = std::accumulate(weight.mShape.begin(), weight.mShape.end(), 1,
                                std::multiplies<int>());

    mWeights.resize(size);
    auto rawptr = (float *)weight.mRawptr;
    memcpy(mWeights.data(), rawptr, size * sizeof(float));
  }

  std::vector<int> mAxis, mWeightShape;
  std::vector<float> mWeights;
};
REGISTER_LAYER(Gather);

class RangeGather : public LayerBase {
public:
  RangeGather(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto npastptr = (int64_t *)mInputs[2];
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    int n_past = *npastptr;
    if (yshape.n == 3) {
      for (size_t ib = 0; ib < yshape.ptr[0]; ib++) {
        for (int i = 0; i < yshape.ptr[1]; i++) {
          std::memcpy(yptr + ib * yshape.ptr[1] * yshape.ptr[2] +
                          i * yshape.ptr[2],
                      mWeights.data() + (n_past + i) * yshape.ptr[2],
                      yshape.ptr[2] * sizeof(float));
        }
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    auto &weight = tensors[2];
    auto size = std::accumulate(weight.mShape.begin(), weight.mShape.end(), 1,
                                std::multiplies<int>());

    mWeights.resize(size);
    auto rawptr = (float *)weight.mRawptr;
    memcpy(mWeights.data(), rawptr, size * sizeof(float));
  }

  std::vector<float> mWeights;
};
REGISTER_LAYER(RangeGather);

class Add : public LayerBase {
public:
  Add(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto x1ptr = mWeights.data();
    std::vector<int> tmpWS;
    if (mInputs.size() == 1) {
      tmpWS.resize(xshape.n);
      int i = 0;
      int padding = xshape.n - mWShape.size();
      for (; i < padding; i++) {
        tmpWS[i] = 1;
      }
      for (; i < xshape.n; i++) {
        tmpWS[i] = mWShape[i - padding];
      }
    }

    tensor_shape_t wtmp{tmpWS.size(), tmpWS.data()};
    auto x1shape = wtmp;
    if (mInputs.size() > 1) {
      x1ptr = (float *)mInputs[1];
      x1shape = mIShapes[1];
    }
    auto vol0 = shape_volume(mIShapes[0]);
    auto vol1 = shape_volume(mIShapes[1]);

    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    if (vol0 == vol1) {
      for (size_t i = 0; i < vol0; i++) {
        yptr[i] = xptr[i] + x1ptr[i];
      }
      return;
    }
    if (xshape.n == 3 && x1shape.n == xshape.n) {
      for (size_t ib = 0; ib < yshape.ptr[0]; ib++) {
        int x0_offset =
            xshape.ptr[0] == 1 ? 0 : ib * xshape.ptr[1] * xshape.ptr[2];
        int x1_offset =
            x1shape.ptr[0] == 1 ? 0 : ib * x1shape.ptr[1] * x1shape.ptr[2];
        for (int i = 0; i < yshape.ptr[1]; i++) {
          int x0_offset_1 =
              xshape.ptr[1] == 1 ? x0_offset : x0_offset + i * xshape.ptr[2];
          int x1_offset_1 =
              x1shape.ptr[1] == 1 ? x1_offset : x1_offset + i * x1shape.ptr[2];
          for (int j = 0; j < yshape.ptr[2]; j++) {
            int x0_offset_2 =
                xshape.ptr[2] == 1 ? x0_offset_1 : x0_offset_1 + j;
            int x1_offset_2 =
                x1shape.ptr[2] == 1 ? x1_offset_1 : x1_offset_1 + j;
            yptr[ib * yshape.ptr[1] * yshape.ptr[2] + i * yshape.ptr[2] + j] =
                xptr[x0_offset_2] + x1ptr[x1_offset_2];
          }
        }
      }
    } else if (xshape.n == 4) {
      for (int i = 0; i < xshape.ptr[0]; i++) {
        for (int j = 0; j < xshape.ptr[1]; j++) {
          for (int k = 0; k < xshape.ptr[2]; k++) {
            for (int l = 0; l < xshape.ptr[3]; l++) {
              auto offset = i * xshape.ptr[1] * xshape.ptr[2] * xshape.ptr[3] +
                            j * xshape.ptr[2] * xshape.ptr[3] +
                            k * xshape.ptr[3] + l;
              yptr[offset] = xptr[offset] + x1ptr[offset];
            }
          }
        }
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    if (tensors.size() == 0) {
      return;
    }
    auto w = tensors[0];
    mWShape = w.mShape;
    auto size = std::accumulate(w.mShape.begin(), w.mShape.end(), 1,
                                std::multiplies<int>());
    mWeights.resize(size);
    std::memcpy(mWeights.data(), w.mRawptr, size * sizeof(float));
  }
  std::vector<float> mWeights;
  std::vector<int> mWShape;
};
REGISTER_LAYER(Add);

class Mul : public LayerBase {
public:
  Mul(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto x1ptr = mWeights.data();
    std::vector<int> tmpWS;
    if (mInputs.size() == 1) {
      tmpWS.resize(xshape.n);
      int i = 0;
      int padding = xshape.n - mWShape.size();
      for (; i < padding; i++) {
        tmpWS[i] = 1;
      }
      for (; i < xshape.n; i++) {
        tmpWS[i] = mWShape[i - padding];
      }
    }

    tensor_shape_t wtmp{tmpWS.size(), tmpWS.data()};
    auto x1shape = wtmp;
    if (mInputs.size() > 1) {
      x1ptr = (float *)mInputs[1];
      x1shape = mIShapes[1];
    }
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    if (xshape.n == 3 && x1shape.n == xshape.n) {
      for (size_t ib = 0; ib < yshape.ptr[0]; ib++) {
        int x0_offset =
            xshape.ptr[0] == 1 ? 0 : ib * xshape.ptr[1] * xshape.ptr[2];
        int x1_offset =
            x1shape.ptr[0] == 1 ? 0 : ib * x1shape.ptr[1] * x1shape.ptr[2];
        for (int i = 0; i < yshape.ptr[1]; i++) {
          int x0_offset_1 =
              xshape.ptr[1] == 1 ? x0_offset : x0_offset + i * xshape.ptr[2];
          int x1_offset_1 =
              x1shape.ptr[1] == 1 ? x1_offset : x1_offset + i * x1shape.ptr[2];
          for (int j = 0; j < yshape.ptr[2]; j++) {
            int x0_offset_2 =
                xshape.ptr[2] == 1 ? x0_offset_1 : x0_offset_1 + j;
            int x1_offset_2 =
                x1shape.ptr[2] == 1 ? x1_offset_1 : x1_offset_1 + j;
            yptr[ib * yshape.ptr[1] * yshape.ptr[2] + i * yshape.ptr[2] + j] =
                xptr[x0_offset_2] * x1ptr[x1_offset_2];
          }
        }
      }
    } else if (xshape.n == 4) {
      for (int i = 0; i < xshape.ptr[0]; i++) {
        for (int j = 0; j < xshape.ptr[1]; j++) {
          for (int k = 0; k < xshape.ptr[2]; k++) {
            for (int l = 0; l < xshape.ptr[3]; l++) {
              auto offset = i * xshape.ptr[1] * xshape.ptr[2] * xshape.ptr[3] +
                            j * xshape.ptr[2] * xshape.ptr[3] +
                            k * xshape.ptr[3] + l;
              yptr[offset] = xptr[offset] * x1ptr[offset];
            }
          }
        }
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    if (tensors.size() == 0) {
      return;
    }
    auto w = tensors[0];
    mWShape = w.mShape;
    auto size = std::accumulate(w.mShape.begin(), w.mShape.end(), 1,
                                std::multiplies<int>());
    mWeights.resize(size);
    std::memcpy(mWeights.data(), w.mRawptr, size * sizeof(float));
  }
  std::vector<float> mWeights;
  std::vector<int> mWShape;
};
REGISTER_LAYER(Mul);

class Mad : public LayerBase {
public:
  Mad(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto yptr = (float *)mOutputs[0];
    auto xshape = mIShapes[0];
    std::vector<int> tmpWS;
    if (mInputs.size() != 1) {
      return;
    }
    int nfea = xshape.ptr[xshape.n - 1];
    if (mWShape[0] != nfea) {
      return;
    }
    auto batch = shape_volume(xshape.ptr, xshape.n - 1);
    for (size_t i = 0; i < batch; i++) {
      for (size_t j = 0; j < nfea; j++) {
        yptr[i * nfea + j] = xptr[i * nfea + j] * mAlpha[j] + mBeta[j];
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    if (tensors.size() != 2) {
      return;
    }
    auto w = tensors[0];
    mWShape = w.mShape;
    auto size = std::accumulate(w.mShape.begin(), w.mShape.end(), 1,
                                std::multiplies<int>());
    mAlpha.resize(size);
    mBeta.resize(size);
    std::memcpy(mAlpha.data(), w.mRawptr, size * sizeof(float));
    std::memcpy(mBeta.data(), tensors[1].mRawptr, size * sizeof(float));
  }
  std::vector<float> mAlpha, mBeta;
  std::vector<int> mWShape;
};
REGISTER_LAYER(Mad);

class Layernrom : public LayerBase {
public:
  Layernrom(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {
    mReduceMean0_axes = attrs["ReduceMean0_axes"].mInts;
    mReduceMean4_axes = attrs["ReduceMean3_axes"].mInts;
  }

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    if (mReduceMean0_axes[0] == mReduceMean4_axes[0] &&
        mReduceMean0_axes[0] == -1) {
      if (yshape.n == 3) {
        for (size_t ib = 0; ib < yshape.ptr[0]; ib++) {
          int offset0 = ib * yshape.ptr[1] * yshape.ptr[2];
          for (int i = 0; i < yshape.ptr[1]; i++) {
            int offset1 = offset0 + i * yshape.ptr[2];
            float xsum = 0.f, x2sum = 0.f;
            for (int j = 0; j < yshape.ptr[2]; j++) {
              int offset2 = offset1 + j;
              xsum += xptr[offset2];
              x2sum += xptr[offset2] * xptr[offset2];
            }
            float aveg = xsum / yshape.ptr[2];
            float var = (x2sum / yshape.ptr[2] - aveg * aveg);
            float svar = std::sqrtf(var + mEpsilon);
            for (int j = 0; j < yshape.ptr[2]; j++) {
              int offset2 = offset1 + j;
              yptr[offset2] = (xptr[offset2] - aveg) / svar;
            }
          }
        }
      }
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    if (tensors.size() >= 2) {
      mEpsilon = *(float *)tensors[1].mRawptr;
    }
  }
  std::vector<int> mReduceMean0_axes, mReduceMean4_axes;
  float mEpsilon = 0.f;
};
REGISTER_LAYER(Layernrom);

static void transpose0213(const float *src, float *dst, int d0, int d1, int d2,
                          int d3, int ldsrc) {
  int sstep0 = d1 * ldsrc;
  int dstep0 = d1 * d2 * d3;
  int sstep1 = ldsrc;
  int dstep1 = d1 * d3;
  int sstep2 = d3;
  int dstep2 = d3;
  for (int i = 0; i < d0; i++) {
    for (int j = 0; j < d1; j++) {
      for (int k = 0; k < d2; k++) {
        for (int l = 0; l < d3; l++) {
          dst[i * dstep0 + k * dstep1 + j * dstep2 + l] =
              src[i * sstep0 + j * sstep1 + k * sstep2 + l];
        }
      }
    }
  }
}

static void transpose0213(const float *src, float *dst, int d0, int d1, int d2,
                          int d3, int lds0, int lds1, int lds2, int ldd0,
                          int ldd1, int ldd2) {
  for (int i = 0; i < d0; i++) {
    for (int j = 0; j < d1; j++) {
      for (int k = 0; k < d2; k++) {
        for (int l = 0; l < d3; l++) {
          dst[i * ldd0 + k * ldd1 + j * ldd2 + l] =
              src[i * lds0 + j * lds1 + k * lds2 + l];
        }
      }
    }
  }
}
class MHA : public LayerBase {
public:
  MHA(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto kvptr = (float *)mOutputs[0]; // kv cache
    auto yptr = (float *)mOutputs[1];
    auto kvshape = mOShapes[0];
    auto yshape = mOShapes[1];
    auto qkvptr = (float *)mInputs[0];
    auto qkvshape = mIShapes[0];
    int hiddensize = qkvshape.ptr[1] / 3;
    int headsize = kvshape.ptr[4];
    int headnum = kvshape.ptr[2];
    if (qkvshape.ptr[1] % 3 != 0) {
      return;
    }
    int seq_len = kvshape.ptr[3];
    int batch = kvshape.ptr[1];
    std::vector<float> tempbuf(seq_len * hiddensize * 4);
    auto qptr = qkvptr;
    auto kptr = qptr + hiddensize;
    auto vptr = kptr + hiddensize;
    auto kTptr = kvptr;
    auto vTptr = kTptr + batch * seq_len * hiddensize;
    transpose0213(kptr, kTptr, batch, seq_len, headnum, headsize,
                  hiddensize * 3);
    transpose0213(vptr, vTptr, batch, seq_len, headnum, headsize,
                  hiddensize * 3);
    for (int i = 0; i < batch; i++) {

      auto qTptr = tempbuf.data();
      transpose0213(qptr + i * seq_len * hiddensize * 3, qTptr, 1, seq_len,
                    headnum, headsize, hiddensize * 3);
      auto mm0ptr = qTptr + seq_len * hiddensize;
      auto mm1ptr = mm0ptr + seq_len * hiddensize;
      auto kTbatchptr = kTptr + i * seq_len * hiddensize;
      auto vTbatchptr = vTptr + i * seq_len * hiddensize;
      auto maskptr = mMaskArray.data();
      int mask_maxlen = mMaskDim[1];
      // first matmul
      float rscale = 1.f / mMMScale;
      for (size_t j = 0; j < headnum; j++) {
        for (size_t im = 0; im < seq_len; im++) {
          float expsum = 0.f;
          float maxexp = 0.f;
          for (size_t in = 0; in < seq_len; in++) {
            float tmp = 0.f;
            for (size_t ik = 0; ik < headsize; ik++) {
              tmp += qTptr[j * seq_len * headsize + im * headsize + ik] *
                     kTbatchptr[j * seq_len * headsize + in * headsize + ik];
            }
            auto m = maskptr[im * mask_maxlen + in];
            mm0ptr[j * seq_len * seq_len + im * seq_len + in] =
                m == 1.f ? tmp * rscale : -10000.f;
            maxexp = mm0ptr[j * seq_len * seq_len + im * seq_len + in] > maxexp
                         ? mm0ptr[j * seq_len * seq_len + im * seq_len + in]
                         : maxexp;
          }
          for (size_t in = 0; in < seq_len; in++) {
            mm0ptr[j * seq_len * seq_len + im * seq_len + in] = std::expf(
                mm0ptr[j * seq_len * seq_len + im * seq_len + in] - maxexp);
            expsum += mm0ptr[j * seq_len * seq_len + im * seq_len + in];
          }
          for (size_t in = 0; in < seq_len; in++) {
            mm0ptr[j * seq_len * seq_len + im * seq_len + in] /= expsum;
          }
          for (size_t in = 0; in < headsize; in++) {
            float tmp = 0.f;
            for (size_t ik = 0; ik < seq_len; ik++) {
              tmp += mm0ptr[j * seq_len * seq_len + im * seq_len + ik] *
                     vTbatchptr[j * seq_len * headsize + ik * headsize + in];
            }
            mm1ptr[j * seq_len * headsize + im * headsize + in] = tmp;
          }
        }
      }
      auto ybatchptr = yptr + i * seq_len * hiddensize;
      transpose0213(mm1ptr, ybatchptr, 1, headnum, seq_len, headsize,
                    headsize * seq_len);
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    mMMScale = *(float *)tensors[6].mRawptr;
    auto &mask = tensors[0];
    mMaskDim.resize(2);
    mMaskDim[0] = mask.mShape[2];
    mMaskDim[1] = mask.mShape[3];
    // mMaskSubOp0 = *(float *)tensors[7].mRawptr;
    // mMaskMulOp1 = *(float *)tensors[8].mRawptr;
    mMaskArray.resize(mMaskDim[0] * mMaskDim[1]);
    memcpy(mMaskArray.data(), mask.mRawptr,
           mMaskArray.size() * sizeof(uint8_t));
  }
  float mMMScale = 1.f;
  float mMaskSubOp0 = 1.f;
  float mMaskMulOp1 = 1000.f;
  std::vector<uint8_t> mMaskArray;
  std::vector<int> mMaskDim;
};
REGISTER_LAYER(MHA);

class MHA_KVcache : public LayerBase {
public:
  MHA_KVcache(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto kvptr = (float *)mOutputs[0]; // kv cache
    auto yptr = (float *)mOutputs[1];
    auto kvshape = mOShapes[0];
    auto yshape = mOShapes[1];
    auto qkvptr = (float *)mInputs[0];
    auto npastptr = (int64_t *)mInputs[4];
    auto qkvshape = mIShapes[0];
    int hiddensize = qkvshape.ptr[1] / 3;
    int headsize = kvshape.ptr[4];
    int headnum = kvshape.ptr[2];
    int seq_len = kvshape.ptr[3];
    int batch = kvshape.ptr[1];
    int mask_maxlen = mMaskDim[1];
    int max_seq = mask_maxlen;
    auto qptr = qkvptr;
    auto kptr = qptr + hiddensize;
    auto vptr = kptr + hiddensize;
    auto kTptr = kvptr;
    auto vTptr = kTptr + batch * max_seq * hiddensize;
    int n_past = *npastptr;
    int len_total = n_past + seq_len;
    auto tempbuf = new float[seq_len * hiddensize * 2 + seq_len * len_total];
    transpose0213(kptr, kTptr + n_past * headsize, batch, seq_len, headnum,
                  headsize, seq_len * hiddensize * 3, hiddensize * 3, headsize,
                  headnum * max_seq * headsize, max_seq * headsize, headsize);
    transpose0213(vptr, vTptr + n_past * headsize, batch, seq_len, headnum,
                  headsize, seq_len * hiddensize * 3, hiddensize * 3, headsize,
                  headnum * max_seq * headsize, max_seq * headsize, headsize);
    for (int i = 0; i < batch; i++) {
      auto qTptr = tempbuf;                       // seq*hiddensize
      auto mm0ptr = qTptr + seq_len * hiddensize; // seq*len_total
      auto mm1ptr = mm0ptr + seq_len * len_total; // seq*hiddensize
      transpose0213(qptr + i * seq_len * hiddensize * 3, qTptr, 1, seq_len,
                    headnum, headsize, hiddensize * 3);

      auto kTbatchptr = kTptr + i * max_seq * hiddensize;
      auto vTbatchptr = vTptr + i * max_seq * hiddensize;
      auto maskptr = mMaskArray.data();
      // first matmul
      float rscale = 1.f / mMMScale;
      for (size_t j = 0; j < headnum; j++) {
        for (size_t im = 0; im < seq_len; im++) {
          float expsum = 0.f;
          float maxexp = 0.f;
          for (size_t in = 0; in < len_total; in++) {
            float tmp = 0.f;
            for (size_t ik = 0; ik < headsize; ik++) {
              tmp += qTptr[j * seq_len * headsize + im * headsize + ik] *
                     kTbatchptr[j * max_seq * headsize + in * headsize + ik];
            }
            auto m = maskptr[(im + n_past) * mask_maxlen + in];
            mm0ptr[im * len_total + in] = m == 1.f ? tmp * rscale : -10000.f;
            maxexp = mm0ptr[im * len_total + in] > maxexp
                         ? mm0ptr[im * len_total + in]
                         : maxexp;
          }
          for (size_t in = 0; in < len_total; in++) {
            mm0ptr[im * len_total + in] =
                std::expf(mm0ptr[im * len_total + in] - maxexp);
            expsum += mm0ptr[im * len_total + in];
          }
          for (size_t in = 0; in < len_total; in++) {
            mm0ptr[im * len_total + in] /= expsum;
          }
          for (size_t in = 0; in < headsize; in++) {
            float tmp = 0.f;
            for (size_t ik = 0; ik < len_total; ik++) {
              tmp += mm0ptr[im * len_total + ik] *
                     vTbatchptr[j * max_seq * headsize + ik * headsize + in];
            }
            mm1ptr[j * seq_len * headsize + im * headsize + in] = tmp;
          }
        }
      }
      auto ybatchptr = yptr + i * seq_len * hiddensize;
      transpose0213(mm1ptr, ybatchptr, 1, headnum, seq_len, headsize,
                    headsize * seq_len);
    }
    delete[] tempbuf;
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    mMMScale = *(float *)tensors[6].mRawptr;
    auto &mask = tensors[0];
    mMaskDim.resize(2);
    mMaskDim[0] = mask.mShape[2];
    mMaskDim[1] = mask.mShape[3];
    // mMaskSubOp0 = *(float *)tensors[7].mRawptr;
    // mMaskMulOp1 = *(float *)tensors[8].mRawptr;
    mMaskArray.resize(mMaskDim[0] * mMaskDim[1]);
    memcpy(mMaskArray.data(), mask.mRawptr,
           mMaskArray.size() * sizeof(uint8_t));
  }
  float mMMScale = 1.f;
  float mMaskSubOp0 = 1.f;
  float mMaskMulOp1 = 1000.f;
  std::vector<uint8_t> mMaskArray;
  std::vector<int> mMaskDim;
};
REGISTER_LAYER(MHA_KVcache);

class Gelu : public LayerBase {
public:
  Gelu(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto yptr = (float *)mOutputs[0];
    auto yshape = mOShapes[0];
    auto xvol = shape_volume(mIShapes[0]);
    auto yvol = shape_volume(mOShapes[0]);
    if (xvol != yvol) {
      return;
    }
    for (size_t i = 0; i < xvol; i++) {
      float x = xptr[i];
      float tmp = x * x * x * mParams[2] + x;
      tmp *= mParams[3];
      tmp = tanh(tmp) + mParams[4];
      tmp *= x * mParams[0];
      yptr[i] = tmp;
    }
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    for (size_t i = 0; i < 5; i++) {
      mParams[i] = *(float *)tensors[i].mRawptr;
    }
  }
  float mParams[5];
};
REGISTER_LAYER(Gelu);

class MatMul : public LayerBase {
public:
  using tSGEMM_AVX2 = jblas::wrapper::gemm::LauncherBase<
      JblasAVX2, jblas::gemm::SCoreRowNAvx2<24, 4>,
      jblas::prologue_a::gemm::ActivationBase,
      jblas::prologue_b::gemm::WeightPack,
      jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
  MatMul(const char *_name, attr_map_t &attrs) : LayerBase(_name, attrs) {}

  void forward(std::vector<void *> &mInputs,
               std::vector<tensor_shape_t> &mIShapes,
               std::vector<void *> &mOutputs,
               std::vector<tensor_shape_t> &mOShapes) override {
    auto xptr = (float *)mInputs[0];
    auto xshape = mIShapes[0];
    auto M = shape_volume(xshape.ptr, xshape.n - 1);
    auto yptr = (float *)mOutputs[0];
    if (mInputs.size() == 2) {
      return;
    }
    jblas::utils::GemmProblem gp(1, M, N, K);
    tSGEMM_AVX2::Param args{gp, {xptr, K}, {NULL, 0, &mPackedB}, {yptr, N}};
    using tParallel =
        jblas::parallel::gemm::SchedulerBase<tSGEMM_AVX2::GemmCore>;
    jblas::parallel::GemmRun<tParallel>(mKernelAVX2, args,
                                        ne_jblas::ne_threading::get());
  }

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors) override {
    auto &weight = tensors[0];
    K = weight.mShape[0];
    N = weight.mShape[1];
    mPackedB = mKernelAVX2.mProB.createStorage(N, K);
    mBuf.resize(mPackedB.mSize);
    mPackedB.assign(mBuf.data());
    auto rawptr = (float *)weight.mRawptr;
    mKernelAVX2.mProB.packWeight(N, K, {rawptr, N, &mPackedB},
                                 ne_jblas::ne_threading::get());
  }
  tSGEMM_AVX2 mKernelAVX2;
  tSGEMM_AVX2::PrologueB::StorageType mPackedB{0};
  int N, K;
  jblas::utils::avector<int8_t> mBuf;
};
REGISTER_LAYER(MatMul);
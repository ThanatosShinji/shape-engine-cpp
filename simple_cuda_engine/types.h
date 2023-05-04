#pragma once

#include <stdexcept>
#include <numeric>
#include <array>
#include <string>
#include <vector>
#include <stdarg.h>
#include <cmath>
#include <cuda.h>
#include "cuda_runtime_api.h"
#include "cuda_fp16.h"

struct KernelParam2D {
  int paddings[2];
  int kernels[2];
  int dilations[2];
  int strides[2];
};

enum class PoolingMode: int
{
  max,
  average,
  min,
};

enum class ResizeMode :int
{
  nearest,
  linear,
  cubic,
};

enum class ElewiseOp :int
{
  Add,
  Sub,
  Mul,
  Div,
};

enum class ElewiseType : int
{
  Scalar,
  Vector,
  Tensor,
};

enum class ReduceOp :int
{
  Mean,
  Max,
};

enum class ResizeCoordinateMode :int
{
  asymmetric,
  pytorch_half_pixel,
  half_pixel,
  align_corners,
  tf_half_pixel_for_nn,
};


enum class CudaKernelActivationType :int
{
  Linear=0,
  Relu=1,
  LeakyRelu=2,
  Clip=3,
  PRelu=4,
  Sigmoid=5,
  Swish=6,
  HardSigmoid=7,
  HardSwish=8,
  Tanh=9,
};

struct ActivationParam
{
  ActivationParam()
    :type(CudaKernelActivationType::Linear)
    ,alpha(0.f)
    ,beta(0.f)
  {

  }
  CudaKernelActivationType type;
  float alpha, beta;
};

enum class CudaPaddingMode : int
{
  Zero=0,
  Edge=1,
};

#define ACT_COMPARE_RET(str,ACT) \
if(!strcmp(str,#ACT))\
{\
  return CudaKernelActivationType::ACT;\
}

CudaKernelActivationType __inline ActivationString2Enum(const char* str)
{
  ACT_COMPARE_RET(str, Relu);
  ACT_COMPARE_RET(str, LeakyRelu);
  ACT_COMPARE_RET(str, Clip);
  ACT_COMPARE_RET(str, PRelu);
  ACT_COMPARE_RET(str, Sigmoid);
  ACT_COMPARE_RET(str, Swish);
  ACT_COMPARE_RET(str, HardSigmoid);
  ACT_COMPARE_RET(str, HardSwish);
  ACT_COMPARE_RET(str, Tanh);
  return CudaKernelActivationType::Linear;
}

#define PADDING_COMPARE_RET(str,P) \
if(!strcmp(str,#P))\
{\
  return CudaPaddingMode::P;\
}

CudaPaddingMode __inline PaddingModeString2Enum(const char* str)
{
  PADDING_COMPARE_RET(str, Edge);
  return CudaPaddingMode::Zero;
}

#define RESIZE_COMPARE_RET(str,STR) \
if(!strcmp(str,#STR))\
{\
  return ResizeMode::STR;\
}

ResizeMode __inline ResizeModeString2Enum(const char* str)
{
  RESIZE_COMPARE_RET(str, linear);
  RESIZE_COMPARE_RET(str, cubic);
  return ResizeMode::nearest;
}

#define REDUCE_COMPARE_RET(str,STR) \
if(!strcmp(str,#STR))\
{\
  return ReduceOp::STR;\
}

ReduceOp __inline ReduceOpString2Enum(const char* str)
{
  REDUCE_COMPARE_RET(str, Max);
  return ReduceOp::Mean;
}

#define RESIZECOOR_COMPARE_RET(str,STR) \
if(!strcmp(str,#STR))\
{\
  return ResizeCoordinateMode::STR;\
}

ResizeCoordinateMode __inline ResizeCoordinateString2Enum(const char* str)
{
  RESIZECOOR_COMPARE_RET(str, asymmetric);
  RESIZECOOR_COMPARE_RET(str, pytorch_half_pixel);
  RESIZECOOR_COMPARE_RET(str, half_pixel);
  RESIZECOOR_COMPARE_RET(str, tf_half_pixel_for_nn);
  return ResizeCoordinateMode::align_corners;
}

#define ELEWISE_COMPARE_RET(str,STR) \
if(!strcmp(str,#STR))\
{\
  return ElewiseOp::STR;\
}

ElewiseOp __inline ElementwiseString2Enum(const char* str)
{
  ELEWISE_COMPARE_RET(str, Mul);
  ELEWISE_COMPARE_RET(str, Div);
  ELEWISE_COMPARE_RET(str, Sub);
  return ElewiseOp::Add;
}

static inline int updiv(int a, int b)
{
  return (a + b - 1) / b;
}

static inline int padto(int a, int b)
{
  return updiv(a, b) * b;
}

static inline int padto_le(int a, int b)
{
  return a / b * b;
}
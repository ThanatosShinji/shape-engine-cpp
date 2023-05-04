#pragma once
#include "types.h" 

bool cudaBatchNormalization(const float* src, const float* sm
	, const float* beta, const float* mean, float epsilon, float* dst
	, const int n, const int c, const int h, const int w, ActivationParam actparam
	, void* _stream);


bool cudaPooling2Df(const float* src, const int n, const int c, const int h, const int w
	, float* dst, const int oh, const int ow, PoolingMode mode, KernelParam2D param, void* _stream);

bool cudaGlobalPooling2Df(const float* src, const int n, const int c, const int h, const int w
	, float* dst, PoolingMode mode, void* _stream);


bool cudaConv2D_k1(const float* src, const float* wei, const float* bias
	, const int n, const int c, const int h, const int w, int group
	, float* dst, const int oc, const int oh, const int ow, int stride
  , ActivationParam actparam, void* _stream);
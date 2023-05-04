#pragma once
#include <cmath>
#include <cuda_fp16.h>

static inline __host__ __device__ int Updiv(int a, int b)
{
	return (a + b - 1) / b;
}


template<typename T>
__device__ __inline__ T MaxOp(const T& a, const T& b)
{
	return a > b ? a : b;
}

template<typename T>
__device__ __inline__ T MinOp(const T& a, const T& b)
{
	return a > b ? b : a;
}

template<typename T>
__device__ __inline__ T PowOp(const T& a, const T& b)
{
	return pow(a, b);
}

template<>
__device__ __inline__ half PowOp(const half& a, const half& b)
{
	return pow(float(a), float(b));
}

template<typename T>
__device__ __inline__  T ReluOp(const T src)
{
	return MaxOp((T)(0), src);
}

template<typename T>
__device__ __inline__ T ClipOp(const T src, const T l_coef, const T h_coef)
{
	T t = MinOp(h_coef, src);
	return MaxOp((T)(l_coef), t);
}

template<typename T>
__device__ __inline__ T PReluOp(const T src, const T pr_coef)
{
	return src < (T)(0) ? src * pr_coef : src;
}

template<typename T>
__device__ __inline__  T LeakyReluOp(const T src, const T lr_coef)
{
	return src < (T)(0) ? src * lr_coef : src;
}

template<typename T>
__device__ __inline__ T SigmoidOp(const T src)
{
	T one = (T)(1);
	return one / (one + exp(-src));
}

template<typename T>
__device__ __inline__ T SwishOp(const T src) {
	return src * SigmoidOp(src);
}

template<typename T>
__device__ __inline__ T HardSigmoidOp(const T src, const T alpha, const T beta)
{
	return MaxOp((T)(0), MinOp((T)(1), alpha * src + (T)(beta)));
}

template<typename T>
__device__ __inline__ T HardSwishOp(const T src, const T alpha, const T beta)
{
	return src * HardSigmoidOp(src, alpha, beta);
}

template<typename T>
__device__ __inline__ T TanhOp(const T src)
{
	return tanh(float(src));
}

//template<>
//__device__ __inline__ half TanhOp(const half src)
//{
//    half one = half(1);
//    half e = hexp(half(2)*src);
//    return __hdiv(e - one, e + one);
//}


template<typename T>
__inline__ __device__ T LinearAct(T& t)
{
	return t;
}

template<typename T>
__inline__ __device__ T ReluAct(T& t)
{
	return t >= T(0) ? t : T(0);
}

template<typename T>
__inline__ __device__ T LeakyReluAct(T& t, T param)
{
	return t >= T(0) ? t : t * param;
}


template<typename T>
__device__ __forceinline__ void DoActivation(T* const tset, const int size, CudaKernelActivationType acttype, const T alpha, const T beta)
{
	switch (acttype)
	{
	case CudaKernelActivationType::Relu:
		for (int l = 0; l < size; l++)
		{
			tset[l] = ReluOp(tset[l]);
		}
		break;
	case CudaKernelActivationType::LeakyRelu:
		for (int l = 0; l < size; l++)
		{
			tset[l] = LeakyReluOp(tset[l], alpha);
		}
		break;
	case CudaKernelActivationType::Clip:
		for (int l = 0; l < size; l++)
		{
			tset[l] = ClipOp(tset[l], alpha, beta);
		}
		break;
	case CudaKernelActivationType::Sigmoid:
		for (int l = 0; l < size; l++)
		{
			tset[l] = SigmoidOp(tset[l]);
		}
		break;
	case CudaKernelActivationType::Swish:
		for (int l = 0; l < size; l++)
		{
			tset[l] = SwishOp(tset[l]);
		}
		break;
	case CudaKernelActivationType::HardSigmoid:
		for (int l = 0; l < size; l++)
		{
			tset[l] = HardSigmoidOp(tset[l], alpha, beta);
		}
		break;
	case CudaKernelActivationType::HardSwish:
		for (int l = 0; l < size; l++)
		{
			tset[l] = HardSwishOp(tset[l], alpha, beta);
		}
		break;
	case CudaKernelActivationType::Tanh:
		for (int l = 0; l < size; l++)
		{
			tset[l] = TanhOp(tset[l]);
		}
		break;
	case CudaKernelActivationType::Linear:
	case CudaKernelActivationType::PRelu:
	default:
		break;
	}
}

template<int ele2copy, typename _CT, typename _T>
__device__ __inline__ void DeviceTypeMemcopy(const _T* src, _T* dst)
{
	constexpr int elepercopy = sizeof(_CT) / sizeof(_T);
	constexpr int copyloop = ele2copy / elepercopy;
#pragma unroll
	for (size_t i = 0; i < copyloop; i++)
	{
		*(_CT*)(dst + i * elepercopy) = *(_CT*)(src + i * elepercopy);
	}
}
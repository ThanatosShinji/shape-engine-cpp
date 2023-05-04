#include "kernel.h"
#include "kernel_func.cuh"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

template<int WTile>
__global__ void cuda_batchnorm_activation_nchw(const float* src, const float* sm
	, const float* beta, const float* mean, const float epsilon, float* dst
	, const int n, const int c, const int h, const int w, ActivationParam actparam)
{
	const int w_threads = Updiv(w, WTile);
	const int MaxThread = n * h * w_threads * c;
	int gpos = blockIdx.x * blockDim.x + threadIdx.x;
	if (gpos >= MaxThread)
	{
		return;
	}
	int tw = (gpos % w_threads) * WTile;
	gpos /= w_threads;
	int th = gpos % h;
	gpos /= h;
	int tc = gpos % c;
	gpos /= c;
	int tn = gpos;
	int offset = tn * c * h * w + tc * h * w + th * w + tw;

	auto _sm = sm[tc];
	auto sv = beta[tc];
	auto _mean = mean[tc];
	float x[WTile], tmp[WTile];
	DeviceTypeMemcopy<4, float4, float>(&src[offset], x);
#pragma unroll
	for (int i = 0; i < WTile; i++)
	{
		tmp[i] = _sm * (x[i] - _mean) + sv;
		tmp[i] = actparam.type == CudaKernelActivationType::Relu ? ReluOp(tmp[i]) : tmp[i];
	}
	DeviceTypeMemcopy<WTile, float4, float>(tmp, &dst[offset]);
}

template <unsigned nthds_per_cta>
//__launch_bounds__(nthds_per_cta)
__global__ void cuda_cta_batchnorm_activation_nchw(
	const float* __restrict__ src, const float* sm
	, const float* beta, const float* mean, const float epsilon, float* dst
	, const int n, const int c, const int h, const int w, ActivationParam actparam)
{
	const int MaxThread = n * h * w * c;
	for (int bpos = blockIdx.x;; bpos += gridDim.x)
	{
		int gpos = bpos * blockDim.x + threadIdx.x;
		if (gpos > MaxThread)
		{
			break;
		}
		int tw = (gpos % w);
		gpos /= w;
		int th = gpos % h;
		gpos /= h;
		int tc = gpos % c;
		gpos /= c;
		int tn = gpos;
		int offset = tn * c * h * w + tc * h * w + th * w + tw;
		auto _sm = sm[tc];
		auto sv = beta[tc];
		auto _mean = mean[tc];
		float x, tmp;
		x = src[offset];
		tmp = _sm * (x - _mean) + sv;
		tmp = actparam.type == CudaKernelActivationType::Relu ? ReluOp(tmp) : tmp;
		dst[offset] = tmp;
	}
}


bool cudaBatchNormalization(const float* src, const float* sm
	, const float* beta, const float* mean, float epsilon, float* dst
	, const int n, const int c, const int h, const int w, ActivationParam actparam
	, void* _stream)
{
	auto stream = (cudaStream_t)_stream;
	constexpr int WTile = 4;
	if (w % WTile == 0)
	{
		constexpr int threadcount = 512;
		dim3 threads{ threadcount };
		int const w_threads = Updiv(w, WTile);
		dim3 grid(Updiv(n * h * w_threads * c, threads.x));
		cuda_batchnorm_activation_nchw<WTile> << <grid, threads, 0, stream >> > (src, sm, beta, mean, epsilon, dst, n, c, h, w, actparam);
	}
	else
	{
		const int BS = 256;
		const int GS = 256;
		cuda_cta_batchnorm_activation_nchw<BS> << <GS, BS, 0, stream >> > (src, sm, beta, mean, epsilon, dst, n, c, h, w, actparam);
	}
	return true;
}

__global__ void cuda_maxpool_nchw(const float* src, float* dst
	, const int n, const int c, const int h, const int w
	, const int oh, const int ow, KernelParam2D param)
{
	typedef float copytype_t;
	const int MaxThread = n * c * oh * ow;
	int gpos = blockIdx.x * blockDim.x + threadIdx.x;
	int o_offset = gpos;
	if (gpos >= MaxThread)
	{
		return;
	}
	int tw = gpos % ow;
	gpos /= ow;
	int th = gpos % oh;
	gpos /= oh;
	int tc = gpos % c;
	gpos /= c;
	int tn = gpos;

	auto srcptr = src + tn * h * w * c + tc * h * w;
	auto tarptr = dst + o_offset;
	float tmp = 0.f;
	for (int iy = 0; iy < param.kernels[0]; iy++)
	{
		auto srcy = th * param.strides[0] + iy - param.paddings[0];
		srcy = MinOp(srcy, h - 1);
		srcy = MaxOp(srcy, 0);
		for (int ix = 0; ix < param.kernels[1]; ix++)
		{
			auto srcx = tw * param.strides[1] + ix - param.paddings[1];
			srcx = MinOp(srcx, w - 1);
			srcx = MaxOp(srcx, 0);
			tmp = MaxOp(tmp, *(srcptr + srcy * w + srcx));
		}
	}
	*tarptr = tmp;
}

bool cudaPooling2Df(const float* src, const int n, const int c, const int h, const int w
	, float* dst, const int oh, const int ow, PoolingMode mode, KernelParam2D param, void* _stream)
{
	constexpr int threadcount = 512;
	dim3 threads{ threadcount };
	dim3 grid(updiv(n * oh * ow * c, threads.x));
	auto stream = (cudaStream_t)_stream;
	switch (mode)
	{
	case PoolingMode::max:
		cuda_maxpool_nchw << <grid, threads, 0, stream >> > (src, dst, n, c, h, w, oh, ow, param);
		return true;
	default:
		break;
	}
	return false;
}

__global__ void cuda_globalaveragepool_nchw(const float* src, float* dst
	, const int n, const int c, const int h, const int w)
{
	typedef float copytype_t;
	const int MaxThread = n * c * 1 * 1;
	int gpos = blockIdx.x * blockDim.x + threadIdx.x;
	int o_offset = gpos;
	if (gpos >= MaxThread)
	{
		return;
	}
	int tw = gpos % 1;
	gpos /= 1;
	int th = gpos % 1;
	gpos /= 1;
	int tc = gpos % c;
	gpos /= c;
	int tn = gpos;

	auto srcptr = src + tn * h * w * c + tc * h * w;
	auto tarptr = dst + o_offset;
	float tmp = 0.f;
	for (int iy = 0; iy < h; iy++)
	{
		auto srcy = iy;
		for (int ix = 0; ix < w; ix++)
		{
			auto srcx = ix;
			tmp = tmp + *(srcptr + srcy * w + srcx);
		}
	}
	*tarptr = tmp / (h * w);
}

bool cudaGlobalPooling2Df(const float* src, const int n, const int c, const int h, const int w
	, float* dst, PoolingMode mode, void* _stream)
{
	constexpr int threadcount = 512;
	dim3 threads{ threadcount };
	dim3 grid(updiv(n * 1 * 1 * c, threads.x));
	auto stream = (cudaStream_t)_stream;
	switch (mode)
	{
	case PoolingMode::average:
		cuda_globalaveragepool_nchw << <grid, threads, 0, stream >> > (src, dst, n, c, h, w);
		return true;
	default:
		break;
	}
	return false;
}

__global__ void cuda_fconv2d_nhwc_oc32_k1(const float* src, float* dst
	, const float* __restrict__ wei, const float* bias
	, const int n, const int c, const int h, const int w
	, const int oc, const int oh, const int ow, const int stride, ActivationParam actparam)
{
	int constexpr THREAD_OC = 4;
	int constexpr THREAD_W = 4;
	int constexpr BLOCK_OC = 32;
	int constexpr OC_THREADS = BLOCK_OC / THREAD_OC;
	int constexpr WARP_W = 32 / OC_THREADS * THREAD_W;
	int constexpr WARP_NUM = 8;
	int constexpr BLOCK_W = WARP_W * WARP_NUM;
	int constexpr BLOCK_THREADS = WARP_NUM * 32;
	if (blockDim.x != BLOCK_THREADS)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0)
		{
			printf("INVALID THREAD CONFIG \n");
		}
		return;
	}
	int laneID = threadIdx.x % 32;
	int warpID = threadIdx.x / 32;
	int blkpos = blockIdx.x * BLOCK_W;
	int warpw_in_blk = warpID * WARP_W;
	int warpw = blkpos + warpw_in_blk;
	int thdw_in_warp = laneID / OC_THREADS * THREAD_W;
	int thdw = warpw + thdw_in_warp;
	int thdoc = (laneID % OC_THREADS) * THREAD_OC;
	auto tarptr = dst + thdw * stride + thdoc;
	auto wptr = wei + thdoc;
	int constexpr TileK = 32;
	int constexpr SHARE_STRIDE = TileK;
	__shared__ float tmpA[BLOCK_W * SHARE_STRIDE];
	float tmp[THREAD_W * THREAD_OC];
	int valid_warp_compute = warpw < n * oh * ow;
	if (!valid_warp_compute)
	{
		return;
	}
	if (bias != NULL)
	{
		for (int i = 0; i < THREAD_W; i++)
		{
			for (int j = 0; j < THREAD_OC; j++)
			{
				tmp[i * THREAD_OC + j] = bias[thdoc + j];
			}
		}
	}
	else
	{
		for (int i = 0; i < THREAD_W * THREAD_OC; i++)
		{
			tmp[i] = 0.f;
		}
	}

	int valid_thdw = thdw + THREAD_W <= n * oh * ow ? THREAD_W : n * oh * ow - thdw;
	auto shA_warp = tmpA + warpw_in_blk * SHARE_STRIDE;
	for (int ic = 0; ic < c; ic += TileK)
	{
		if (c - ic == 16)
		{
#pragma unroll
			for (int i = 0; i < WARP_W; i += 2)
			{
				int cpyw = i + laneID / 16;
				int cpyc = laneID % 16;
				bool flag = warpw + cpyw >= n * h * w;
				shA_warp[cpyw * SHARE_STRIDE + cpyc] = flag ? 0.f : *(src
					+ (warpw + cpyw) * c + cpyc + ic);
			}
		}
		else
		{
#pragma unroll
			for (int i = 0; i < WARP_W; i += 1)
			{
				int cpyw = i;
				int cpyc = laneID;
				bool flag = warpw + cpyw >= n * h * w;
				shA_warp[cpyw * SHARE_STRIDE + cpyc] = flag ? 0.f : *(src
					+ (warpw + cpyw) * c + cpyc + ic);
			}
		}
#if 1
		for (int icc = 0; icc < TileK; icc += 4)
		{
			if (ic + icc >= c)
			{
				break;
			}
			float4 xval[THREAD_W];
			for (int iw = 0; iw < THREAD_W; iw++)
			{
				xval[iw] = *(float4*)&shA_warp[(thdw_in_warp + iw) * SHARE_STRIDE + icc];
			}
			for (int i = 0; i < 4; i++)
			{
				float wval[THREAD_OC];
				DeviceTypeMemcopy<THREAD_OC, float4, float>(&wptr[(ic + icc + i) * oc], wval);
				for (int itw = 0; itw < THREAD_W; itw++)
				{
					for (int itc = 0; itc < THREAD_OC; itc++)
					{
						tmp[itw * THREAD_OC + itc] += wval[itc] * ((float*)&xval[itw])[i];
					}
				}
			}
		}
#else
		for (int icc = 0; icc < TileK; icc += 1)
		{
			float xval[THREAD_W];
			for (int iw = 0; iw < THREAD_W; iw++)
			{
				xval[iw] = shA_warp[(thdw_in_warp + iw) * SHARE_STRIDE + icc];
			}
			float wval[THREAD_OC];
			DeviceTypeMemcopy<THREAD_OC, float4, float>(&wptr[(ic + icc) * oc], wval);
			for (int itw = 0; itw < THREAD_W; itw++)
			{
				for (int itc = 0; itc < THREAD_OC; itc++)
				{
					tmp[itw * THREAD_OC + itc] += wval[itc] * xval[itw];
				}
			}
		}
#endif
	}
	DoActivation(tmp, THREAD_OC * THREAD_W, actparam.type, actparam.alpha, actparam.beta);
	for (int itw = 0; itw < valid_thdw; itw++)
	{
		DeviceTypeMemcopy<THREAD_OC, float4, float>(tmp + itw * THREAD_OC, tarptr + itw * stride);
	}
}


bool cudaConv2D_k1(const float* src, const float* wei, const float* bias
	, const int n, const int c, const int h, const int w, int group
	, float* dst, const int oc, const int oh, const int ow, int stride
	, ActivationParam actparam, void* _stream)
{
	constexpr int threadcount = 256;
	dim3 threads{ threadcount };
	int constexpr BLOCK_W = 128;
	const int BlockCount = Updiv(n * oh * ow, BLOCK_W);
	auto stream = (cudaStream_t)_stream;
	dim3 grid(BlockCount);
	cuda_fconv2d_nhwc_oc32_k1 << <grid, threads, 0, stream >> > (src, dst, wei, bias, n, c, h, w, oc, oh, ow, stride, actparam);
	return true;
}
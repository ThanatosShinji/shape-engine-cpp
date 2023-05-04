#include "cuda_engine.h"
#include <numeric>
#include <cublasLt.h>
#include "cudnn.h"
#include "kernel.h"

using namespace cuda_engine;
using namespace onnx_tool::shape_engine;

std::vector<LayerCreator*> LayerFactory::mCreatetors;

class Conv :public LayerBase
{
public:
	Conv(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{
		if (mHandle == NULL)
		{
			cudnnCreate(&mHandle);
		}
		cudnnCreateConvolutionDescriptor(&convDesc);
		cudnnCreateTensorDescriptor(&xDesc);
		cudnnCreateTensorDescriptor(&zDesc);
		cudnnCreateTensorDescriptor(&yDesc);
		cudnnCreateTensorDescriptor(&biasDesc);
		cudnnCreateFilterDescriptor(&wDesc);
		cudnnCreateActivationDescriptor(&activationDesc);

		mDilations = attrs["dilations"].mInts;
		mGroup = attrs["group"].mInts;
		mKernel = attrs["kernel_shape"].mInts;
		mPads = attrs["pads"].mInts;
		mStrides = attrs["strides"].mInts;
		cudnnSetActivationDescriptor(activationDesc, cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY, cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN, 1.f);
		alpha1 = 1.f;
		alpha2 = 0.f;
		if (attrs["postop_count"].mInts.size() > 0)
		{
			mPostopCount = attrs["postop_count"].mInts[0];
			for (int i = 0; i < mPostopCount; i++)
			{
				std::string key = std::string("postop_") + std::to_string(i);
				mPostops.push_back(attrs[key].mStrings[0]);
				if (attrs[key].mStrings[0] == "Relu")
				{
					cudnnSetActivationDescriptor(activationDesc, cudnnActivationMode_t::CUDNN_ACTIVATION_RELU, cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN, 1.f);
					mAct.type = CudaKernelActivationType::Relu;
				}
				if (attrs[key].mStrings[0] == "Add")
				{
					alpha2 = 1.f;
				}
			}
		}
		else
		{
			mPostopCount = 0;
		}

	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, cudaStream_t _stream) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto yptr = (float*)mOutputs[0];
		auto x1ptr = yptr;
		if (mInputs.size() > 1)
		{
			x1ptr = (float*)mInputs[1];
		}
		auto yshape = mOShapes[0];
		int batch = xshape.ptr[0];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		int oh = yshape.ptr[2];
		int ow = yshape.ptr[3];
		if (mKernel[0]==1&&mOC%32==0)
		{
			cudaConv2D_k1(xptr, mDWeights.data(), mDBias.data(), batch, mIC, h, w, 1, yptr, mOC, oh, ow, mOC, mAct, _stream);
			return;
		}
		cudnnSetTensor4dDescriptor(xDesc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, batch, mIC, h, w);
		cudnnSetTensor4dDescriptor(yDesc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, batch, mOC, oh, ow);
		cudnnSetTensor4dDescriptor(zDesc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, batch, mOC, oh, ow);
		auto ret = cudnnConvolutionBiasActivationForward(mHandle, &alpha1, xDesc, xptr, wDesc, mDWeights.data(), convDesc, cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
			, NULL, 0, &alpha2, zDesc, x1ptr, biasDesc, mDBias.data(), activationDesc, yDesc, yptr);
		if (ret)
		{
			printf("%d\n", ret);
		}
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
		auto& weight = tensors[0];
		mIC = weight.mShape[1];
		mOC = weight.mShape[0];
		auto newsize = mIC * mOC * mKernel[0] * mKernel[1];
		auto rawptr = (float*)weight.mRawptr;
		cudnnSetFilter4dDescriptor(wDesc, cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, mOC, mIC, mKernel[0], mKernel[1]);
		cudnnSetTensor4dDescriptor(biasDesc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, mOC, 1, 1);
		cudnnSetConvolutionNdDescriptor(convDesc, 2, mPads.data(), mStrides.data(), mDilations.data(), cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION, cudnnDataType_t::CUDNN_DATA_FLOAT);
		mDWeights.resize(newsize);
		cudaMemcpy(mDWeights.data(), rawptr, mDWeights.size() * 4, cudaMemcpyHostToDevice);
		mDBias.resize(mOC);
		if (tensors.size() > 1)
		{
			auto& bias = tensors[1];
			cudaMemcpy(mDBias.data(), bias.mRawptr, mDBias.size() * 4, cudaMemcpyHostToDevice);
		}
		else
		{
			cudaMemset(mDBias.data(), 0, mDBias.size() * 4);
		}
#if 0
		if (mName == "resnetv24_stage1_conv3_fwd")
		{
			int h = 56, w = 56;
			int oh = 56, ow = 56;
			cudnnSetFilter4dDescriptor(wDesc, cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, mOC, mIC, mKernel[0], mKernel[1]);
			cudnnSetTensor4dDescriptor(biasDesc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, mOC, 1, 1);
			cudnnSetTensor4dDescriptor(xDesc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, mIC, h, w);
			cudnnSetTensor4dDescriptor(yDesc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, mOC, oh, ow);
			cudnnSetTensor4dDescriptor(zDesc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, mOC, oh, ow);
			cu_vector<float> tmpX;
			tmpX.resize(mIC * h * w);
			cu_vector<float> tmpY;
			tmpY.resize(mOC * oh * ow);
			int maxcount = 0;
			int retcount = 0;
			cudnnConvolutionFwdAlgoPerf_t perf[25];
			workSpaceSize = 1024LL * 1024 * 1024;
			workspace.resize(workSpaceSize);
			cudnnGetConvolutionForwardAlgorithmMaxCount(mHandle, &maxcount);
			cudnnFindConvolutionForwardAlgorithmEx(mHandle, xDesc, tmpX.data(), wDesc, mDWeights.data(), convDesc
				, yDesc, tmpY.data(), maxcount, &retcount,perf,workspace.data(),workSpaceSize);
			printf("%d\n", retcount);
		}
#endif

	}

	int mIC, mOC, mPostopCount;
	std::vector<int> mDilations, mGroup, mKernel, mPads, mStrides;
	cu_vector<float> mDWeights, mDBias;
	cudnnTensorDescriptor_t xDesc, zDesc, biasDesc, yDesc;
	cudnnFilterDescriptor_t wDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnActivationDescriptor_t activationDesc;
	std::vector<std::string> mPostops;
	size_t workSpaceSize;
	float alpha1, alpha2;
	cu_vector<int8_t> workspace;
	static cudnnHandle_t mHandle;
	ActivationParam mAct;
};
cudnnHandle_t Conv::mHandle = NULL;
REGISTER_LAYER(Conv);

class BatchNormalization :public LayerBase
{
public:
	BatchNormalization(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{
		mSpatial = attrs["spatial"].mInts;
		mEpsilon = attrs["epsilon"].mFloats;
		mMomentum = attrs["momentum"].mFloats;
		if (attrs["postop_count"].mInts.size() > 0)
		{
			for (int i = 0; i < attrs["postop_count"].mInts[0]; i++)
			{
				std::string key = std::string("postop_") + std::to_string(i);
				mPostops.push_back(attrs[key].mStrings[0]);
				if (attrs[key].mStrings[0] == "Relu")
				{
					mAct.type = CudaKernelActivationType::Relu;
				}

			}
		}
	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, cudaStream_t _stream) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		int batch = xshape.ptr[0];
		int cn = xshape.ptr[1];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		cudaBatchNormalization(xptr, mSM.data(), mBeta.data(), mMean.data(), mEpsilon[0], yptr, batch, cn, h, w, mAct, _stream);
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
		auto& gamma = tensors[0];
		auto& beta = tensors[1];
		auto& mean = tensors[2];
		auto& var = tensors[3];
		mOC = gamma.mShape[0];
		mBeta.resize(mOC);
		mMean.resize(mOC);
		cudaMemcpy(mBeta.data(), beta.mRawptr, mOC * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(mMean.data(), mean.mRawptr, mOC * 4, cudaMemcpyHostToDevice);
		mSM.resize(mOC);
		std::vector<float> tmpSM(mOC);
		for (int j = 0; j < mOC; j++)
		{
			auto sqrt_variance = std::sqrt(((float*)var.mRawptr)[j] + mEpsilon[0]);
			tmpSM[j] = ((float*)gamma.mRawptr)[j] / sqrt_variance;
		}
		cudaMemcpy(mSM.data(), tmpSM.data(), tmpSM.size() * 4, cudaMemcpyHostToDevice);
	}

	int mOC;
	std::vector<int> mSpatial;
	std::vector<float> mEpsilon, mMomentum;
	cu_vector<float> mSM, mBeta, mMean;
	std::vector<std::string> mPostops;
	ActivationParam mAct;
};
REGISTER_LAYER(BatchNormalization);

class MaxPool :public LayerBase
{
public:
	MaxPool(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{
		mKernel = attrs["kernel_shape"].mInts;
		mPads = attrs["pads"].mInts;
		mStrides = attrs["strides"].mInts;
		mParam.strides[0] = mStrides[0];
		mParam.strides[1] = mStrides[1];
		mParam.dilations[0] = 1;
		mParam.dilations[1] = 1;
		mParam.kernels[0] = mKernel[0];
		mParam.kernels[1] = mKernel[0];
		mParam.paddings[0] = mPads[0];
		mParam.paddings[1] = mPads[0];
	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, cudaStream_t _stream) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto xsize = shape_size(xshape);
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		auto ysize = shape_size(yshape);
		int batch = xshape.ptr[0];
		int cn = xshape.ptr[1];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		int oh = yshape.ptr[2];
		int ow = yshape.ptr[3];
		cudaPooling2Df(xptr, batch, cn, h, w, yptr, oh, ow, PoolingMode::max, mParam, _stream);
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
	}
	std::vector<int> mKernel, mPads, mStrides;
	KernelParam2D mParam;
};
REGISTER_LAYER(MaxPool);

class GlobalAveragePool :public LayerBase
{
public:
	GlobalAveragePool(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{
	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, cudaStream_t _stream) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto xsize = shape_size(xshape);
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		auto ysize = shape_size(yshape);
		int batch = xshape.ptr[0];
		int cn = xshape.ptr[1];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		cudaGlobalPooling2Df(xptr, batch, cn, h, w, yptr, PoolingMode::average, _stream);
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
	}
};
REGISTER_LAYER(GlobalAveragePool);

#include "cublas_v2.h"
inline void checkCublasStatus(cublasStatus_t status) {
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("cuBLAS API failed with status %d\n", status);
		throw std::logic_error("cuBLAS API failed");
	}
}
class Gemm :public LayerBase
{
public:
	Gemm(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{
		if (mHandle == NULL)
		{
			cublasLtCreate(&mHandle);
		}
		mAlpha = attrs["alpha"].mFloats;
		mBeta = attrs["beta"].mFloats;
		mTransA = attrs["transA"].mInts;
		mTransB = attrs["transB"].mInts;

	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, cudaStream_t _stream) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		M = xshape.ptr[0];
		if (mTransA[0])
		{
			M = xshape.ptr[1];
		}
		auto xsize = shape_size(xshape);
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		auto ysize = shape_size(yshape);
		if (!(mTransA[0] == 0 && mTransB[0] == 1))
		{
			//implement TransA=0 TransB=1 only
			return;
		}
#if 0
		auto xhost = cuda2Vec(xptr, xsize);
		std::vector<float> yhost(ysize, 0.f);
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				auto tmp = 0.f;
				if (mBias.size())
				{
					tmp = mBias[j];
				}
				for (int k = 0; k < K; k++)
				{
					tmp += xhost[i * K + k] * mWeights[j * K + k];
				}
				yhost[i * N + j] = tmp;
	}
}
		cudaMemcpy(yptr, yhost.data(), yhost.size() * 4, cudaMemcpyHostToDevice);
#else
		int lda = tA == cublasOperation_t::CUBLAS_OP_N ? M : K;
		int ldb = tB == cublasOperation_t::CUBLAS_OP_N ? K : N;
		cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, tA == CUBLAS_OP_N ? M : K, tA == CUBLAS_OP_N ? K : M, lda);
		cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, tB == CUBLAS_OP_N ? K : N, tB == CUBLAS_OP_N ? N : K, ldb);
		cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, 1);
		cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, M, N, M);
		//cublasSgemm_v2(mHandle, tA, tB, M, N, K, &alpha, xptr,lda,(const float*)mDWeights.data(),ldb
		//	,&beta,)
		cublasLtMatmulPreference_t preference = NULL;

		int returnedResults = 0;
		cublasLtMatmulHeuristicResult_t heuristicResult = {};
		checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
		void* workspace = NULL;
		size_t workspaceSize = 0;

		checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

		// we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
		// is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
		checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(mHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

		checkCublasStatus(cublasLtMatmul(mHandle,
			operationDesc,
			&alpha,
			xptr,
			Adesc,
			mDWeights.data(),
			Bdesc,
			&beta,
			mDBias.data(),
			Cdesc,
			yptr,
			Ddesc,
			&heuristicResult.algo,
			workspace,
			workspaceSize,
			_stream));
#endif
}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
		auto& weight = tensors[0];
		if (mTransB[0])
		{
			K = weight.mShape[1];
			N = weight.mShape[0];
		}
		else
		{
			K = weight.mShape[0];
			N = weight.mShape[1];
		}

		auto newsize = N * K;
		mWeights.resize(newsize);
		auto rawptr = (float*)weight.mRawptr;
		memcpy(mWeights.data(), rawptr, newsize * sizeof(float));
		mDWeights.resize(newsize);
		cudaMemcpy(mDWeights.data(), mWeights.data(), newsize * 4, cudaMemcpyHostToDevice);
		mDBias.resize(N);
		if (tensors.size() > 1)
		{
			auto& bias = tensors[1];
			mBias.resize(N);
			memcpy(mBias.data(), bias.mRawptr, N * sizeof(float));
			beta = 1.f;
			cudaMemcpy(mDBias.data(), mBias.data(), N * 4, cudaMemcpyHostToDevice);
		}
		tA = mTransA[0] ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T;
		tB = mTransB[0] ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T;

		cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
		cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
		cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));

	}
	cublasOperation_t tA, tB;
	int N, K, M;
	float alpha = 1.f, beta = 0.f;
	std::vector<float> mAlpha, mBeta;
	std::vector<int> mTransA, mTransB;
	std::vector<float> mWeights;
	cu_vector<float> mDWeights, mDBias;
	std::vector<float> mBias;
	static cublasLtHandle_t mHandle;
	cublasLtMatmulDesc_t operationDesc = NULL;
	cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
	cublasLtMatmulPreference_t preference = NULL;
};
cublasLtHandle_t Gemm::mHandle = NULL;
REGISTER_LAYER(Gemm);

// simple_inference_engine.cpp: 目标的源文件。
//

#include "simple_inference_engine.h"
#include <numeric>

using namespace simple_inference_engine_f32;

std::vector<LayerCreator*> LayerFactory::mCreatetors;

class Conv :public LayerBase
{
public:
	Conv(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{
		mDilations = attrs["dilations"].mInts;
		mGroup = attrs["group"].mInts;
		mKernel = attrs["kernel_shape"].mInts;
		mPads = attrs["pads"].mInts;
		mStrides = attrs["strides"].mInts;
		if (attrs["postop_count"].mInts.size() > 0)
		{
			mPostopCount = attrs["postop_count"].mInts[0];
			for (int i = 0; i < mPostopCount; i++)
			{
				std::string key = std::string("postop_") + std::to_string(i);
				mPostops.push_back(attrs[key].mStrings[0]);
			}
		}
		else
		{
			mPostopCount = 0;
		}
	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		int batch = xshape.ptr[0];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		int oh = yshape.ptr[2];
		int ow = yshape.ptr[3];

		for (int i = 0; i < batch; i++)
		{
			auto xbatchptr = xptr + i * batch * mIC * h * w;
			auto ybatchptr = yptr + i * batch * mOC * oh * ow;
			for (int j = 0; j < mOC; j++)
			{
				float bias = 0.f;
				if (mBias.size())
				{
					bias = mBias[j];
				}
				for (int k = 0; k < oh; k++)
				{
					auto yhptr = ybatchptr + j * oh * ow + k * ow;
					for (int l = 0; l < ow; l++)
					{
						float tmp = bias;
						for (int iy = 0; iy < mKernel[0]; iy++)
						{
							auto srcy = k * mStrides[0] + iy - mPads[0];
							if (srcy < 0 || srcy >= h)
							{
								continue;
							}
							auto xhptr = xbatchptr + srcy * w;
							for (int ix = 0; ix < mKernel[1]; ix++)
							{
								auto srcx = l * mStrides[1] + ix - mPads[1];
								if (srcx >= 0 && srcx < w)
								{
									for (int icn = 0; icn < mIC; icn++)
									{
										auto xval = xhptr[srcx + icn * h * w];
										auto wval = mWeights[j * mKernel[0] * mKernel[1] * mIC + iy * mKernel[1] * mIC + ix * mIC + icn];
										tmp += xval * wval;
									}

								}
							}
						}
						for (int ipost = 0; ipost < mPostopCount; ipost++)
						{
							if (std::strcmp(mPostops[ipost].c_str(), "Relu") == 0)
							{
								tmp = std::max(0.f, tmp);
							}
							if (std::strcmp(mPostops[ipost].c_str(), "Add") == 0)
							{
								auto post1ptr = (float*)mInputs[1];
								tmp += post1ptr[i * batch * mOC * oh * ow + j * oh * ow + k * ow + l];
							}
						}
						yhptr[l] = tmp;
					}
				}
			}
		}
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
		auto& weight = tensors[0];
		mIC = weight.mShape[1];
		mOC = weight.mShape[0];
		auto newsize = mIC * mOC * mKernel[0] * mKernel[1];
		mWeights.resize(newsize);
		auto rawptr = (float*)weight.mRawptr;
		//tranpose from Oc*Ic*Ky*Kx to Oc*Ky*Kx*Ic
		for (int io = 0; io < mOC; io++)
		{
			for (int ii = 0; ii < mIC; ii++)
			{
				for (int iky = 0; iky < mKernel[0]; iky++)
				{
					for (int ikx = 0; ikx < mKernel[1]; ikx++)
					{
						mWeights[io * mKernel[0] * mKernel[1] * mIC + iky * mKernel[1] * mIC + ikx * mIC + ii] =
							*(rawptr + io * mKernel[0] * mKernel[1] * mIC + ii * mKernel[0] * mKernel[1] + iky * mKernel[1] + ikx);
					}
				}
			}
		}
		if (tensors.size() > 1)
		{
			auto& bias = tensors[1];
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
			}
		}
	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		int batch = xshape.ptr[0];
		int cn = xshape.ptr[1];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		for (int i = 0; i < batch; i++)
		{
			for (int j = 0; j < cn; j++)
			{
				auto sqrt_variance = std::sqrtf(varptr[j] + mEpsilon[0]);
				auto sm = gammaptr[j] / sqrt_variance;
				auto sv = betaptr[j];
				auto mean = meanptr[j];
				for (int k = 0; k < h; k++)
				{
					for (int l = 0; l < w; l++)
					{
						auto offset = i * cn * h * w + j * h * w + k * w + l;
						yptr[offset] = sm * (xptr[offset] - mean) + sv;
						for (int ipost = 0; ipost < mPostops.size(); ipost++)
						{
							if (std::strcmp(mPostops[ipost].c_str(), "Relu") == 0)
							{
								yptr[offset] = std::max(0.f, yptr[offset]);
							}
						}
					}
				}
			}
		}

	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
		auto& gamma = tensors[0];
		auto& beta = tensors[1];
		auto& mean = tensors[2];
		auto& var = tensors[3];
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
	float* gammaptr, * betaptr, * meanptr, * varptr;
	std::vector<float> mWeights;
	std::vector<int> mSpatial;
	std::vector<float> mEpsilon, mMomentum;
	std::vector<std::string> mPostops;
};
REGISTER_LAYER(BatchNormalization);

class Relu :public LayerBase
{
public:
	Relu(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{

	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		for (int i = 0; i < xshape.ptr[0]; i++)
		{
			for (int j = 0; j < xshape.ptr[1]; j++)
			{
				for (int k = 0; k < xshape.ptr[2]; k++)
				{
					for (int l = 0; l < xshape.ptr[3]; l++)
					{
						auto offset = i * xshape.ptr[1] * xshape.ptr[2] * xshape.ptr[3] + j * xshape.ptr[2] * xshape.ptr[3] + k * xshape.ptr[3] + l;
						yptr[offset] = std::max(xptr[offset], 0.f);
					}
				}
			}
		}
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
	}

};
REGISTER_LAYER(Relu);

class Add :public LayerBase
{
public:
	Add(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{

	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto x1ptr = (float*)mInputs[1];
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		for (int i = 0; i < xshape.ptr[0]; i++)
		{
			for (int j = 0; j < xshape.ptr[1]; j++)
			{
				for (int k = 0; k < xshape.ptr[2]; k++)
				{
					for (int l = 0; l < xshape.ptr[3]; l++)
					{
						auto offset = i * xshape.ptr[1] * xshape.ptr[2] * xshape.ptr[3] + j * xshape.ptr[2] * xshape.ptr[3] + k * xshape.ptr[3] + l;
						yptr[offset] = xptr[offset] + x1ptr[offset];
					}
				}
			}
		}
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
	}

};
REGISTER_LAYER(Add);

class Flatten :public LayerBase
{
public:
	Flatten(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{

	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		auto size = 1;
		for (int i = 0; i < xshape.n; i++)
		{
			size *= xshape.ptr[i];
		}
		std::memcpy(yptr, xptr, size * sizeof(float));

	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
	}

};
REGISTER_LAYER(Flatten);

class MaxPool :public LayerBase
{
public:
	MaxPool(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{
		mKernel = attrs["kernel_shape"].mInts;
		mPads = attrs["pads"].mInts;
		mStrides = attrs["strides"].mInts;
	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		int batch = xshape.ptr[0];
		int cn = xshape.ptr[1];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		int oh = yshape.ptr[2];
		int ow = yshape.ptr[3];

		for (int i = 0; i < batch; i++)
		{
			auto xbatchptr = xptr + i * cn * h * w;
			auto ybatchptr = yptr + i * cn * oh * ow;
			for (int j = 0; j < cn; j++)
			{
				auto xcnptr = xbatchptr + j * h * w;
				auto ycnptr = ybatchptr + j * oh * ow;
				for (int k = 0; k < oh; k++)
				{
					auto yhptr = ycnptr + k * ow;
					for (int l = 0; l < ow; l++)
					{
						float tmp = std::numeric_limits<float>::min();
						for (int iy = 0; iy < mKernel[0]; iy++)
						{
							auto srcy = k * mStrides[0] + iy - mPads[0];
							if (srcy < 0 || srcy >= h)
							{
								continue;
							}
							auto xhptr = xcnptr + srcy * w;
							for (int ix = 0; ix < mKernel[1]; ix++)
							{
								auto srcx = l * mStrides[1] + ix - mPads[1];
								if (srcx >= 0 && srcx < w)
								{
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

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
	}
	std::vector<int> mKernel, mPads, mStrides;
};
REGISTER_LAYER(MaxPool);

class GlobalAveragePool :public LayerBase
{
public:
	GlobalAveragePool(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{
	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		int batch = xshape.ptr[0];
		int cn = xshape.ptr[1];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		for (int i = 0; i < batch; i++)
		{
			for (int j = 0; j < cn; j++)
			{
				auto tmp = 0.f;
				for (int k = 0; k < h; k++)
				{
					for (int l = 0; l < w; l++)
					{
						tmp += xptr[i * cn * h * w + j * h * w + k * w + l];
					}
				}
				tmp /= h * w;
				yptr[i * cn + j] = tmp;
			}
		}
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) override
	{
	}
};
REGISTER_LAYER(GlobalAveragePool);

class Gemm :public LayerBase
{
public:
	Gemm(const char* _name, attr_map_t& attrs)
		:LayerBase(_name, attrs)
	{
		mAlpha = attrs["alpha"].mFloats;
		mBeta = attrs["beta"].mFloats;
		mTransA = attrs["transA"].mInts;
		mTransB = attrs["transB"].mInts;
	}

	void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes) override
	{
		auto xptr = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		M = xshape.ptr[0];
		if (mTransA[0])
		{
			M = xshape.ptr[1];
		}
		auto yptr = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		if (!(mTransA[0] == 0 && mTransB[0] == 1))
		{
			//implement TransA=0 TransB=1 only
			return;
		}
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
					tmp += xptr[i * K + k] * mWeights[j * K + k];
				}
				yptr[i * N + j] = tmp;
			}
		}

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
		if (tensors.size() > 1)
		{
			auto& bias = tensors[1];
			mBias.resize(N);
			memcpy(mBias.data(), bias.mRawptr, N * sizeof(float));
		}
	}

	int N, K, M;
	std::vector<float> mAlpha, mBeta;
	std::vector<int> mTransA, mTransB;
	std::vector<float> mWeights;
	std::vector<float> mBias;
};
REGISTER_LAYER(Gemm);

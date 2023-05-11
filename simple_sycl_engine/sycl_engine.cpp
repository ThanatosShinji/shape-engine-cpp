#include "sycl_engine.h"
#ifdef WITH_MKL
#include <oneapi/mkl.hpp>
#endif

using namespace sycl_engine;

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
		mHasRelu = false;
		mHasAdd = false;
		if (attrs["postop_count"].mInts.size() > 0)
		{
			mPostopCount = attrs["postop_count"].mInts[0];
			for (int i = 0; i < mPostopCount; i++)
			{
				std::string key = std::string("postop_") + std::to_string(i);
				mPostops.push_back(attrs[key].mStrings[0]);
				if (attrs[key].mStrings[0] == "Relu") {
					mHasRelu = true;
				}
				if (attrs[key].mStrings[0] == "Add") {
					mHasAdd = true;
				}
			}
		}
		else
		{
			mPostopCount = 0;
		}
	}

	sycl::event forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, sycl::queue& _stream) override
	{
		auto x = (float*)mInputs[0];
		auto& xshape = mIShapes[0];
		auto xsize = shape_size(xshape);
		auto y = (float*)mOutputs[0];
		auto& yshape = mOShapes[0];
		auto ysize = shape_size(yshape);
		int batch = xshape.ptr[0];
		int cn = xshape.ptr[1];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		int oh = yshape.ptr[2];
		int ow = yshape.ptr[3];
		int oc = mOC;
		int ic = mIC;
		sycl::range num_iter{ batch * oc * oh * ow };
		auto biasptr = mDBias.size() > 0 ? mDBias.data() : NULL;
		auto wptr = mDWeights.data();
		int KY = mKernel[0], KX = mKernel[1];
		int SY = mStrides[0], SX = mStrides[1];
		int PY = mPads[0], PX = mPads[1];
		auto src1ptr = mHasAdd ? (float*)mInputs[1] : NULL;
		bool hasrelu = mHasRelu;

		auto e = _stream.submit([&](sycl::handler& cgh) {
			cgh.parallel_for(num_iter, [=](auto i) {
				int pos = i;
				auto iw = pos % ow;
				pos /= ow;
				auto ih = pos % oh;
				pos /= oh;
				auto ioc = pos % oc;
				pos /= oc;
				auto ibatch = pos;
				auto tmp = biasptr ? biasptr[ioc] : 0.f;
				for (int iy = 0; iy < KY; iy++)
				{
					auto srcy = ih * SY + iy - PY;
					if (srcy < 0 || srcy >= h)
					{
						continue;
					}
					auto xhptr = x + ibatch * ic * h * w + srcy * w;
					for (int ix = 0; ix < KX; ix++)
					{
						auto srcx = iw * SX + ix - PX;
						if (srcx >= 0 && srcx < w)
						{
							for (int icn = 0; icn < ic; icn++)
							{
								auto xval = xhptr[srcx + icn * h * w];
								auto wval = wptr[ioc * KY * KX * ic + iy * KX * ic + ix * ic + icn];
								tmp += xval * wval;
							}

						}
					}
				}
				if (hasrelu)
				{
					tmp = std::max(0.f, tmp);
				}
				auto offset = ibatch * oc * oh * ow + ioc * oh * ow + ih * ow + iw;
				if (src1ptr)
				{
					tmp += src1ptr[offset];
				}
				y[offset] = tmp;
				});
			});
		return e;
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors, sycl::queue& _q) override
	{
		auto& weight = tensors[0];
		mIC = weight.mShape[1];
		mOC = weight.mShape[0];
		auto newsize = mIC * mOC * mKernel[0] * mKernel[1];
		std::vector<float> tmp;
		tmp.resize(newsize);
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
						tmp[io * mKernel[0] * mKernel[1] * mIC + iky * mKernel[1] * mIC + ikx * mIC + ii] =
							*(rawptr + io * mKernel[0] * mKernel[1] * mIC + ii * mKernel[0] * mKernel[1] + iky * mKernel[1] + ikx);
					}
				}
			}
		}
		mDWeights.resize(newsize, _q);
		_q.memcpy(mDWeights.data(), tmp.data(), newsize * 4).wait();
		if (tensors.size() > 1)
		{
			auto& bias = tensors[1];
			mDBias.resize(mOC, _q);
			_q.memcpy(mDBias.data(), bias.mRawptr, bias.mSize).wait();
		}
	}

	int mIC, mOC, mPostopCount;
	std::vector<int> mDilations, mGroup, mKernel, mPads, mStrides;
	std::vector<std::string> mPostops;
	sycl_vector<float> mDWeights, mDBias;
	bool mHasRelu;
	bool mHasAdd;
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
		mHasRelu = false;
		if (attrs["postop_count"].mInts.size() > 0)
		{
			for (int i = 0; i < attrs["postop_count"].mInts[0]; i++)
			{
				std::string key = std::string("postop_") + std::to_string(i);
				mPostops.push_back(attrs[key].mStrings[0]);
				if (attrs[key].mStrings[0] == "Relu")
				{
					mHasRelu = true;
				}

			}
		}
	}

	sycl::event forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, sycl::queue& _stream) override
	{
		auto x = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto y = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		auto xsize = shape_size(xshape);
		auto ysize = shape_size(yshape);
		int batch = xshape.ptr[0];
		int cn = xshape.ptr[1];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		float* gammaptr, * betaptr, * meanptr, * varptr;
		bool hasrelu = mHasRelu;
		sycl::range num_iter{ batch * cn * h * w };
		auto epsilon = mEpsilon[0];
		gammaptr = mDWeights.data();
		betaptr = gammaptr + mOC;
		meanptr = betaptr + mOC;
		varptr = meanptr + mOC;
		auto e = _stream.submit([&](sycl::handler& cgh) {
			cgh.parallel_for(num_iter, [=](auto idx) {
				int pos = idx;
				auto l = pos % w;
				pos /= w;
				auto k = pos % h;
				pos /= h;
				auto j = pos % cn;
				pos /= cn;
				auto i = pos;
				auto sqrt_variance = std::sqrt(varptr[j] + epsilon);
				auto sm = gammaptr[j] / sqrt_variance;
				auto sv = betaptr[j];
				auto mean = meanptr[j];
				auto offset = i * cn * h * w + j * h * w + k * w + l;
				auto tmp = sm * (x[offset] - mean) + sv;
				if (hasrelu)
				{
					tmp = std::max(0.f, tmp);
				}
				y[offset] = tmp;
				});
			});
		return e;
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors, sycl::queue& _q) override
	{
		float* gammaptr, * betaptr, * meanptr, * varptr;
		auto& gamma = tensors[0];
		auto& beta = tensors[1];
		auto& mean = tensors[2];
		auto& var = tensors[3];
		mOC = gamma.mShape[0];
		mDWeights.resize(mOC * 4, _q);
		gammaptr = mDWeights.data();
		betaptr = gammaptr + mOC;
		meanptr = betaptr + mOC;
		varptr = meanptr + mOC;
		_q.memcpy(gammaptr, gamma.mRawptr, mOC * sizeof(float));
		_q.memcpy(betaptr, beta.mRawptr, mOC * sizeof(float));
		_q.memcpy(meanptr, mean.mRawptr, mOC * sizeof(float));
		_q.memcpy(varptr, var.mRawptr, mOC * sizeof(float));
	}

	int mOC;
	std::vector<int> mSpatial;
	std::vector<float> mEpsilon, mMomentum;
	std::vector<std::string> mPostops;
	sycl_vector<float> mDWeights;
	bool mHasRelu;
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
	}

	sycl::event forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, sycl::queue& _stream) override
	{
		auto x = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto y = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		auto xsize = shape_size(xshape);
		auto ysize = shape_size(yshape);
		int batch = xshape.ptr[0];
		int cn = xshape.ptr[1];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		int oh = yshape.ptr[2];
		int ow = yshape.ptr[3];
		sycl::range num_iter{ batch * cn * oh * ow };
		int KY = mKernel[0], KX = mKernel[1];
		int SY = mStrides[0], SX = mStrides[1];
		int PY = mPads[0], PX = mPads[1];
		auto e = _stream.submit([&](sycl::handler& cgh) {
			cgh.parallel_for(num_iter, [=](auto i) {
				int pos = i;
				auto iw = pos % ow;
				pos /= ow;
				auto ih = pos % oh;
				pos /= oh;
				auto icn = pos % cn;
				pos /= cn;
				auto ibatch = pos;
				float tmp = std::numeric_limits<float>::min();
				for (int iy = 0; iy < KY; iy++)
				{
					auto srcy = ih * SY + iy - PY;
					if (srcy < 0 || srcy >= h)
					{
						continue;
					}
					auto xhptr = x + ibatch * cn * h * w + icn * h * w + srcy * w;
					for (int ix = 0; ix < KX; ix++)
					{
						auto srcx = iw * SX + ix - PX;
						if (srcx >= 0 && srcx < w)
						{
							auto xval = xhptr[srcx];
							tmp = std::max(xval, tmp);
						}
					}
				}
				auto offset = ibatch * cn * oh * ow + icn * oh * ow + ih * ow + iw;
				y[offset] = tmp;
				});
			});
		return e;
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors, sycl::queue& _q) override
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

	sycl::event forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, sycl::queue& _stream) override
	{
		auto x = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		auto y = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		auto xsize = shape_size(xshape);
		auto ysize = shape_size(yshape);
		int batch = xshape.ptr[0];
		int cn = xshape.ptr[1];
		int h = xshape.ptr[2];
		int w = xshape.ptr[3];
		int oc = cn;
		sycl::range num_iter{ batch * oc };
		auto e = _stream.submit([&](sycl::handler& cgh) {
			cgh.parallel_for(num_iter, [=](auto i) {
				int pos = i;
				auto ioc = pos % oc;
				pos /= oc;
				auto ibatch = pos;
				auto tmp = 0.f;
				for (int k = 0; k < h; k++)
				{
					for (int l = 0; l < w; l++)
					{
						tmp += x[ibatch * cn * h * w + ioc * h * w + k * w + l];
					}
				}
				tmp /= h * w;
				y[ibatch * cn + ioc] = tmp;
				});
			});
		return e;
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors, sycl::queue& _q) override
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

	sycl::event forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes
		, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, sycl::queue& _stream) override
	{
		auto x = (float*)mInputs[0];
		auto xshape = mIShapes[0];
		M = mTransA[0] ? xshape.ptr[1] : xshape.ptr[0];
		auto y = (float*)mOutputs[0];
		auto yshape = mOShapes[0];
		auto xsize = shape_size(xshape);
		auto ysize = shape_size(yshape);
#if WITH_MKL
		auto colA = mDWeight.data();
		auto colB = x;
		auto colC = y;
		auto colM = N;
		auto colN = M;
		auto colK = K;

		auto coltransA = !mTransB[0] ? oneapi::mkl::transpose::N : oneapi::mkl::transpose::T;
		auto coltransB = !mTransA[0] ? oneapi::mkl::transpose::N : oneapi::mkl::transpose::T;
		auto colLDA = coltransA == oneapi::mkl::transpose::N ? colM : colK;
		auto colLDB = coltransB == oneapi::mkl::transpose::N ? colK : colN;
		auto colLDC = colM;

		auto bptr = mDBias.data();
		auto _N = N;
		auto _M = M;
		oneapi::mkl::blas::gemm(_stream, coltransA, coltransB
			, colM, colN, colK, 1.f, colA, colLDA, colB, colLDB, 0.f, colC, colLDC, oneapi::mkl::blas::compute_mode::any);
		return _stream.submit([&](sycl::handler& h) {
			h.parallel_for(sycl::range<1>{_M* _N}, [=](auto i) {
				auto nidx = i % _N;
				y[i] += bptr[nidx];
				}
			);
			});
#else
		auto bptr = mDWeight.data();
		auto dptr = mDBias.data();
		auto _N = N;
		auto _M = M;
		auto _K = K;
		bool hasbias = mDBias.size();
		if (mTransB[0] && !mTransA[0])
		{
			return _stream.submit([&](sycl::handler& h) {
				h.parallel_for(sycl::range<1>{_M* _N}, [=](auto i) {
					auto in = i % _N;
					auto im = i / _N;
					auto tmp = hasbias ? dptr[in] : 0.f;
					for (int ik = 0; ik < _K; ik++)
					{
						tmp += x[im * _K + ik] * bptr[ik + in * _K];
					}
					y[im * _N + in] = tmp;
					});
				});
		}

#endif

		return sycl::event();
	}

	virtual void setweights(std::vector<onnx_tool::Tensor>& tensors, sycl::queue& _q) override
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
		mDWeight.resize(newsize, _q);
		auto rawptr = (float*)weight.mRawptr;
		memcpy(mWeights.data(), rawptr, newsize * sizeof(float));
		_q.memcpy(mDWeight.data(), rawptr, newsize * sizeof(float)).wait();
		if (tensors.size() > 1)
		{
			auto& bias = tensors[1];
			mBias.resize(N);
			mDBias.resize(N, _q);
			memcpy(mBias.data(), bias.mRawptr, N * sizeof(float));
			_q.memcpy(mDBias.data(), bias.mRawptr, N * sizeof(float)).wait();
		}
	}
	int N, K, M;
	float alpha = 1.f, beta = 0.f;
	std::vector<float> mAlpha, mBeta;
	std::vector<int> mTransA, mTransB;
	std::vector<float> mWeights;
	std::vector<float> mBias;
	sycl_vector<float> mDWeight, mDBias;
};
REGISTER_LAYER(Gemm);
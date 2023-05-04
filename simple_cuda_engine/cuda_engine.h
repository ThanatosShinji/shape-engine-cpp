// simple_inference_engine.h: 目标的头文件。

#pragma once
#include <memory>

#include "shape_engine.h"
#include "graph.h"
#include <chrono>
#include <cuda_runtime.h>

namespace cuda_engine
{
	typedef std::vector<std::pair<std::string, int>> variable_pairs_t;
	typedef onnx_tool::shape_engine::ShapeEngine shape_engine_t;
	typedef onnx_tool::shape_engine::TensorShape tensor_shape_t;
	typedef std::unordered_map<std::string, onnx_tool::Attribute> attr_map_t;


	struct CUDADeleter
	{
		template<class T>
		void operator ()(T* obj)const {
			if (obj)
			{
				cudaFree(obj);
			}
		}
	};

	struct CUDAStreamDeleter
	{
		template<class T>
		void operator ()(T* obj)const {
			if (obj)
			{
				cudaStreamDestroy(obj);
			}
		}
	};

	struct CUDAEventDeleter
	{
		template<class T>
		void operator ()(T* obj)const {
			if (obj)
			{
				cudaEventDestroy(obj);
			}
		}
	};

	template<typename T>
	__inline__ std::vector<T> cuda2Vec(const T* cudaptr, int elecount)
	{
		std::vector<T> tmp(elecount);
		cudaMemcpy(tmp.data(), cudaptr, elecount * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		return tmp;
	}

	typedef std::shared_ptr<CUstream_st> cu_shared_stream_t;
	typedef std::shared_ptr<uint8_t> cu_shared_buffer_t;
	typedef std::shared_ptr<CUevent_st> cu_shared_event_t;

	static	inline cu_shared_buffer_t createCUSmartPtr(uint64_t _size)
	{
		uint8_t* tmp = NULL;
		cudaMalloc((void**)&tmp, (_size));
		return cu_shared_buffer_t(tmp, CUDADeleter());
	}

	static inline cu_shared_stream_t createCUSmartStream()
	{
		cudaStream_t tmp = NULL;
		cudaStreamCreate((cudaStream_t*)&tmp);
		return cu_shared_stream_t(tmp, CUDAStreamDeleter());
	}

	static inline cu_shared_event_t createCUSmartEvent()
	{
		cudaEvent_t tmp = NULL;
		cudaEventCreate((cudaEvent_t*)&tmp);
		return cu_shared_event_t(tmp, CUDAEventDeleter());
	}

	template<typename _T>
	struct cu_vector
	{
		cu_vector()
			:mSize(0)
		{

		}

		void resize(uint64_t _size)
		{
			mSize = _size;
			_T* tmp = NULL;
			cudaMalloc((void**)&tmp, mSize * sizeof(_T));
			mPtr = std::shared_ptr<_T>(tmp, CUDADeleter());
		}

		inline uint64_t size()
		{
			return mSize;
		}

		inline _T* data()
		{
			return mPtr.get(); 
		}

		std::shared_ptr<_T> mPtr;
		uint64_t mSize;
	};


	class TimeRecorder
	{
	public:
		using clk_t = std::chrono::steady_clock;
		TimeRecorder()
		{
			mStart = createCUSmartEvent();
			mEnd = createCUSmartEvent();
		}

		void record_start(void* stream = 0)
		{
			cudaEventRecord(mStart.get(), (cudaStream_t)stream);
		}

		void record_end(void* stream = 0)
		{
			cudaEventRecord(mEnd.get(), (cudaStream_t)stream);
		}

		float elapsed_time()
		{
			float t = 0.f;
			cudaEventElapsedTime(&t, mStart.get(), mEnd.get());
			return t;
		}
		cu_shared_event_t mStart, mEnd;
	};

	class DynamicBindings
	{
	public:
		DynamicBindings(shape_engine_t& _engine)
		{
			mShapeEngine = _engine;
			mStream = createCUSmartStream();
		}

		void reshape(const variable_pairs_t& _variables)
		{
			for (size_t i = 0; i < _variables.size(); i++)
			{
				auto& desc = _variables[i];
				mShapeEngine.update_variable(desc.first.c_str(), desc.second);
			}
			mShapeEngine.update_variables();
		}

		void updateMemory(const variable_pairs_t& _maxvariable)
		{
			reshape(_maxvariable);
			auto dtensor_count = mShapeEngine.mDynamicTensors.size();
			mPtrs.resize(dtensor_count);
			mShapePtr.resize(dtensor_count);
			uint64_t mTotalMem = 0;
			std::vector<uint64_t> memsize;
			for (int i = 0; i < mShapeEngine.mDynamicTensors.size(); i++)
			{
				uint64_t sum = sizeof(float);
				auto tname = mShapeEngine.mDynamicTensors[i].c_str();
				auto n = mShapeEngine.get_tensor_shape_len(tname);
				auto sptr = mShapeEngine.get_tensor_shape_ptr(tname);
				mShapePtr[i] = { n,sptr };
				for (int j = 0; j < n; j++)
				{
					sum *= sptr[j];
				}
				mTotalMem += sum;
				memsize.push_back(sum);
			}
			if (mTotalMem > mBuffer.size())
			{
				mBuffer.resize(mTotalMem);
			}
			auto ptr = mBuffer.data();
			for (int i = 0; i < memsize.size(); i++)
			{
				mPtrs[i] = ptr;
				ptr += memsize[i];
			}
		}

		void sync()
		{
			cudaStreamSynchronize(mStream.get());
		}

		shape_engine_t mShapeEngine;
		std::vector<void*> mPtrs;
		std::vector<tensor_shape_t> mShapePtr;
		cu_vector<uint8_t> mBuffer;
		cu_shared_stream_t mStream;
		std::vector<std::vector<void*>> mLayerInPtr, mLayerOutPtr;
		std::vector<std::vector<tensor_shape_t>> mLayerInShape, mLayerOutShape;
	};

	class LayerBase
	{
	public:
		LayerBase(const char* _name, attr_map_t& attrs) :mName(_name) {};

		virtual ~LayerBase()
		{
		}

		virtual void forward(std::vector<void*>& mInputs, std::vector<tensor_shape_t>& mIShapes, std::vector<void*>& mOutputs, std::vector<tensor_shape_t>& mOShapes, cudaStream_t _stream) = 0;

		virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) = 0;

		TimeRecorder mRecoder;
		std::string mOp;
		std::string mName;
	};

	class LayerCreator
	{
	public:
		virtual const char* getType() = 0;
		virtual LayerBase* createLayer(const char* _name, attr_map_t& _attrs) = 0;
	};

	class LayerFactory
	{
	public:
		static LayerBase* createLayer(const char* _type, const char* _name, attr_map_t& _attrs)
		{
			for (int i = 0; i < mCreatetors.size(); i++)
			{
				auto& creator = mCreatetors[i];
				if (strcmp(creator->getType(), _type) == 0)
				{
					return creator->createLayer(_name, _attrs);
				}
			}
			return nullptr;
		}
		static std::vector<LayerCreator*> mCreatetors;
	};

	template<typename _T>
	struct LayerRegistry
	{
	public:
		LayerRegistry(const char* _type)
		{
			LayerFactory::mCreatetors.push_back(new _T(_type));
		}
	};

	template<typename _T>
	class LayerCreatorT :public LayerCreator
	{
	public:
		LayerCreatorT(const char* _type)
		{
			mType = _type;
		}
		LayerBase* createLayer(const char* _name, attr_map_t& _attrs) override
		{
			auto ptr = new _T(_name, _attrs);
			ptr->mOp = mType;
			return ptr;
		}
		virtual const char* getType() override
		{
			return mType.c_str();
		}
		std::string mType;
	};

#define REGISTER_LAYER(type) static LayerRegistry<LayerCreatorT<type>> _LayerRegistry##type(#type);

	class RuntimeEngine
	{
	public:
		RuntimeEngine()
		{
		}

		~RuntimeEngine()
		{
			for (int i = 0; i < mLayers.size(); i++)
			{
				delete mLayers[i];
			}
		}

		void forward(DynamicBindings* _bindings)
		{
			auto stream = _bindings->mStream.get();
			if (mProfile)
				mRecorder.record_start(stream);
			for (int i = 0; i < mLayers.size(); i++)
			{
				if (mProfile)
					mLayers[i]->mRecoder.record_start(stream);
				mLayers[i]->forward(_bindings->mLayerInPtr[i], _bindings->mLayerInShape[i], _bindings->mLayerOutPtr[i], _bindings->mLayerOutShape[i], stream);
				if (mProfile)
					mLayers[i]->mRecoder.record_end(stream);
			}
			if (mProfile)
				mRecorder.record_end(stream);
		}

		void save_proflie(const char* _file)
		{
			FILE* fp = fopen(_file, "w");
			fprintf(fp, "Name,Type,TimeMs\n");
			for (int i = 0; i < mLayers.size(); i++)
			{
				fprintf(fp, "%s,%s,%f\n", mLayers[i]->mName.c_str(), mLayers[i]->mOp.c_str()
					, mLayers[i]->mRecoder.elapsed_time());
			}
			fprintf(fp, "Total,_,%f\n", mRecorder.elapsed_time());
			fclose(fp);
		}
		std::vector<LayerBase*> mLayers;
		bool mProfile;
		TimeRecorder mRecorder;
	};

	class InferenceContext
	{
	public:
		InferenceContext(onnx_tool::Graph* graph, shape_engine_t& _engine)
		{
			mGraph.reset(graph);
			mShapeEngine = _engine;
			for (size_t i = 0; i < _engine.mDynamicTensors.size(); i++)
			{
				mDynamicIndexMap[_engine.mDynamicTensors[i]] = i;
			}
		}

		RuntimeEngine* createRuntimeEngine()
		{
			auto engine = new RuntimeEngine();
			engine->mLayers.resize(mGraph->mNodes.size());
			for (int i = 0; i < mGraph->mNodes.size(); i++)
			{
				auto& node = mGraph->mNodes[i];
				auto layerptr = LayerFactory::createLayer(node.mOpType.c_str(), node.mName.c_str(), node.mAttributes);
				if (layerptr)
				{
					std::vector<onnx_tool::Tensor> in_stensors;
					for (int iin = 0; iin < node.mInputNames.size(); iin++)
					{
						auto name = node.mInputNames[iin];
						auto iter = mDynamicIndexMap.find(name);
						if (iter == mDynamicIndexMap.end())
						{
							in_stensors.push_back(mGraph->mTensorMap[name]);
						}
					}
					layerptr->setweights(in_stensors);
				}
				engine->mLayers[i] = layerptr;
			}
			return engine;
		}

		DynamicBindings* createDynamicBindings(const variable_pairs_t& max_shape)
		{
			auto ptr = new DynamicBindings(mShapeEngine);
			ptr->updateMemory(max_shape);
			auto layercount = mGraph->mNodes.size();
			ptr->mLayerInPtr.resize(layercount);
			ptr->mLayerOutPtr.resize(layercount);
			ptr->mLayerInShape.resize(layercount);
			ptr->mLayerOutShape.resize(layercount);
			for (int i = 0; i < mGraph->mNodes.size(); i++)
			{
				auto& node = mGraph->mNodes[i];
				auto& in_dptrs = ptr->mLayerInPtr[i];
				auto& out_dptrs = ptr->mLayerOutPtr[i];
				auto& in_shapes = ptr->mLayerInShape[i];
				auto& out_shapes = ptr->mLayerOutShape[i];
				for (int iin = 0; iin < node.mInputNames.size(); iin++)
				{
					auto name = node.mInputNames[iin];
					auto iter = mDynamicIndexMap.find(name);
					if (iter != mDynamicIndexMap.end())
					{
						auto indx = mDynamicIndexMap[name];
						in_dptrs.push_back(ptr->mPtrs[indx]);
						in_shapes.push_back(ptr->mShapePtr[indx]);
					}
				}
				for (int iout = 0; iout < node.mOutputNames.size(); iout++)
				{
					auto name = node.mOutputNames[iout];
					auto iter = mDynamicIndexMap.find(name);
					if (iter != mDynamicIndexMap.end())
					{
						auto indx = mDynamicIndexMap[name];
						out_dptrs.push_back(ptr->mPtrs[indx]);
						out_shapes.push_back(ptr->mShapePtr[indx]);
					}
				}
			}
			return ptr;
		}
		shape_engine_t mShapeEngine;
		std::unordered_map<std::string, int> mDynamicIndexMap;
		std::unique_ptr<onnx_tool::Graph> mGraph;
	};
}
// simple_inference_engine.h: 目标的头文件。

#pragma once
#include <memory>

#include "shape_engine.h"
#include "graph.h"


namespace simple_inference_engine_f32
{
	typedef std::vector<std::pair<std::string, int>> variable_pairs_t;
	typedef onnx_tool::shape_engine::ShapeEngine shape_engine_t;
	typedef std::unordered_map<std::string, onnx_tool::Attribute> attr_map_t;
	struct TensorShape
	{
		int n;
		int* ptr;
	};
	class DynamicBindings
	{
	public:
		DynamicBindings(shape_engine_t& _engine)
		{
			mShapeEngine = _engine;
		}

		void reshape(variable_pairs_t& _variables)
		{
			for (size_t i = 0; i < _variables.size(); i++)
			{
				auto& desc = _variables[i];
				mShapeEngine.update_variable(desc.first.c_str(), desc.second);
			}
			mShapeEngine.update_variables();
		}

		void updateMemory(variable_pairs_t& _maxvariable)
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

		shape_engine_t mShapeEngine;
		std::vector<void*> mPtrs;
		std::vector<TensorShape> mShapePtr;
		std::vector<char> mBuffer;
	};

	class LayerBase
	{
	public:
		LayerBase(const char* _name, attr_map_t& attrs):mName(_name) {};

		virtual ~LayerBase()
		{

		}
		virtual void forward_dynamic()
		{
			for (int i = 0; i < mInputs.size(); i++)
			{
				mInputs[i] = *mInputsRef[i];
				mIShapes[i] = *mIShapesRef[i];
			}
			for (int i = 0; i < mOutputs.size(); i++)
			{
				mOutputs[i] = *mOutputsRef[i];
				mOShapes[i] = *mOShapesRef[i];
			}
			forward();
		}
		virtual void forward() = 0;

		virtual void setweights(std::vector<onnx_tool::Tensor>& tensors) = 0;

		virtual void update()
		{
			auto isize = mInputsRef.size();
			auto osize = mOutputsRef.size();
			mInputs.resize(isize);
			mIShapes.resize(isize);
			mOutputs.resize(osize);
			mOShapes.resize(osize);
		}
		std::string mName;
		std::vector<void**> mInputsRef, mOutputsRef;
		std::vector<TensorShape*> mIShapesRef, mOShapesRef;
	protected:
		
		std::vector<void*> mInputs, mOutputs;
		std::vector<TensorShape> mIShapes, mOShapes;
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
				if (strcmp(creator->getType(),_type)==0)
				{
					return creator->createLayer(_name,_attrs);
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
	class LayerCreatorT:public LayerCreator
	{
	public:
		LayerCreatorT(const char* _type)
		{
			mType = _type;
		}
		LayerBase* createLayer(const char* _name,attr_map_t& _attrs) override
		{
			return new _T(_name, _attrs);
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
		RuntimeEngine(int dsize)
		{
			mMemptrs.resize(dsize);
			mShapes.resize(dsize);
		}

		~RuntimeEngine()
		{
			for (int i = 0; i < mLayers.size(); i++)
			{
				delete mLayers[i];
			}
		}

		void forward_dynamic(DynamicBindings* _bindings)
		{
			for (size_t i = 0; i < _bindings->mPtrs.size(); i++)
			{
				mMemptrs[i] = _bindings->mPtrs[i];
			}
			for (size_t i = 0; i < _bindings->mShapePtr.size(); i++)
			{
				mShapes[i] = _bindings->mShapePtr[i];
			}
			for (int i = 0; i < mLayers.size(); i++)
			{
				mLayers[i]->forward_dynamic();
			}
		}

		void forward()
		{
			for (int i = 0; i < mLayers.size(); i++)
			{
				mLayers[i]->forward();
			}
		}

		std::vector<void*> mMemptrs;
		std::vector<TensorShape> mShapes;
		std::vector<LayerBase*> mLayers;
	};

	class InferenceContext
	{
	public:
		InferenceContext(onnx_tool::Graph* graph,shape_engine_t& _engine)
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
			auto engine = new RuntimeEngine(mShapeEngine.mDynamicTensors.size());
			engine->mLayers.resize(mGraph->mNodes.size());
			for (int i = 0; i < mGraph->mNodes.size(); i++)
			{
				auto& node = mGraph->mNodes[i];
				auto layerptr = LayerFactory::createLayer(node.mOpType.c_str(),node.mName.c_str(),  node.mAttributes);
				if (layerptr)
				{
					std::vector<onnx_tool::Tensor> in_stensors;
					auto& in_dptrs = layerptr->mInputsRef;
					auto& outdptrs = layerptr->mOutputsRef;
					auto& in_shapes = layerptr->mIShapesRef;
					auto& out_shapes = layerptr->mOShapesRef;
					for (int iin = 0; iin < node.mInputNames.size(); iin++)
					{
						auto name = node.mInputNames[iin];
						auto iter = mDynamicIndexMap.find(name);
						if (iter != mDynamicIndexMap.end())
						{
							auto indx = mDynamicIndexMap[name];
							in_dptrs.push_back(&engine->mMemptrs[indx]);
							in_shapes.push_back(&engine->mShapes[indx]);
						}
						else
						{
							in_stensors.push_back(mGraph->mTensorMap[name]);
						}
					}
					for (int iout = 0; iout < node.mOutputNames.size(); iout++)
					{
						auto name = node.mOutputNames[iout];
						auto iter = mDynamicIndexMap.find(name);
						if (iter != mDynamicIndexMap.end())
						{
							auto indx = mDynamicIndexMap[name];
							outdptrs.push_back(&engine->mMemptrs[indx]);
							out_shapes.push_back(&engine->mShapes[indx]);
						}
					}
					layerptr->setweights(in_stensors);
					layerptr->update();
				}
				engine->mLayers[i] = layerptr;
			}
			return engine;
		}

		DynamicBindings* createDynamicBindings()
		{
			return new DynamicBindings(mShapeEngine);
		}
		shape_engine_t mShapeEngine;
		std::unordered_map<std::string, int> mDynamicIndexMap;
		std::unique_ptr<onnx_tool::Graph> mGraph;
	};
}
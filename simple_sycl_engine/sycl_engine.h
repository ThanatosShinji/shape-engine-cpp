#pragma once
#include "graph.h"
#include "shape_engine.h"
#include <chrono>
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <memory>
#include <sycl/sycl.hpp>

namespace sycl_engine {
typedef std::vector<std::pair<std::string, int>> variable_pairs_t;
typedef onnx_tool::shape_engine::ShapeEngine shape_engine_t;
typedef onnx_tool::shape_engine::TensorShape tensor_shape_t;
typedef onnx_tool::MemoryMap memory_map_t;
typedef std::unordered_map<std::string, onnx_tool::Attribute> attr_map_t;
typedef sycl::buffer<uint8_t> sycl_buf_t;

struct SyclDeleter {
  const sycl::queue mQueue;
  SyclDeleter(sycl::queue _q) : mQueue(_q) {}
  template <class T> void operator()(T *obj) const {
    if (obj) {
      sycl::free(obj, mQueue);
    }
  }
};
template <typename _T> struct sycl_vector {
  sycl_vector() : mSize(0) {}

  void resize(uint64_t _size, sycl::queue &_q) {
    mSize = _size;
    _T *tmp = sycl::malloc_device<_T>(_size, _q);
    mPtr = std::shared_ptr<_T>(tmp, SyclDeleter(_q));
  }

  inline uint64_t size() { return mSize; }

  inline _T *data() { return mPtr.get(); }

  std::shared_ptr<_T> mPtr;
  uint64_t mSize;
};

template <typename T>
__inline__ std::vector<T> sycl2Vec(const T *syclptr, int elecount,
                                   sycl::queue q) {
  std::vector<T> tmp(elecount);
  q.memcpy(tmp.data(), syclptr, elecount * sizeof(T)).wait();
  return tmp;
}

class SyclEvent {
public:
  using clk_t = std::chrono::steady_clock;
  SyclEvent() {}

  float elapsed_time() {
    float t = 0.f;
    const auto startKernExecutionTimePoint =
        mEvent
            .get_profiling_info<sycl::info::event_profiling::command_submit>();
    const auto endKernExecutionTimePoint =
        mEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
    t = (endKernExecutionTimePoint - startKernExecutionTimePoint) / 1e6;
    return t;
  }

  float execute_time() {
    float t = 0.f;
    const auto startKernExecutionTimePoint =
        mEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto endKernExecutionTimePoint =
        mEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
    t = (endKernExecutionTimePoint - startKernExecutionTimePoint) / 1e6;
    return t;
  }
  sycl::event mEvent;
};

class TimeRecorder {
public:
  using clk_t = std::chrono::steady_clock;
  TimeRecorder() {}

  void record_start(void *stream = 0) { mStart = clk_t::now(); }

  void record_end(void *stream = 0) { mEnd = clk_t::now(); }

  float elapsed_time() {
    float t = (mEnd - mStart).count() / 1e6;
    return t;
  }
  clk_t::time_point mStart, mEnd;
};

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};
//sycl::queue can get everything you need.
//DNNL makes it complicated
struct FckDNNL {
  sycl::queue *q;
  dnnl::engine *eng;
  dnnl::stream *stream;
};

class DynamicBindings {
public:
  DynamicBindings(shape_engine_t &_engine, bool profile = false)
      : mProfile(profile) {
    mShapeEngine = _engine;
    auto d_selector{sycl::default_selector_v};
    if (profile) {
      sycl::property_list prop = {sycl::property::queue::enable_profiling()};
      mQueue = sycl::queue(d_selector, exception_handler, prop);
    } else {
      mQueue = sycl::queue(d_selector, exception_handler);
    }

    auto ctx = mQueue.get_context();
    auto dev = mQueue.get_device();
    mDNNLEngine = dnnl::sycl_interop::make_engine(dev, ctx);
    mDNNLStream =
        dnnl::sycl_interop::make_stream(mDNNLEngine, mQueue);
    mFck = FckDNNL{&mQueue, &mDNNLEngine, &mDNNLStream};
  }

  ~DynamicBindings() { sync(); }

  bool reshape(const variable_pairs_t &_variables) {
    for (size_t i = 0; i < _variables.size(); i++) {
      auto &desc = _variables[i];
      if (mMaxVariables[desc.first] < desc.second) {
        return false;
      }
      mShapeEngine.update_variable(desc.first.c_str(), desc.second);
    }
    mShapeEngine.update_variables();
    return true;
  }

  void updateMaxShape(const variable_pairs_t &_maxshape) {
    for (size_t i = 0; i < _maxshape.size(); i++) {
      auto &desc = _maxshape[i];
      mMaxVariables[desc.first.c_str()] = desc.second;
    }
  }

  void updateMemory(const variable_pairs_t &_maxvariable) {
    reshape(_maxvariable);
    auto dtensor_count = mShapeEngine.mDynamicTensors.size();
    mShapePtr.resize(dtensor_count);
    uint64_t mTotalMem = 0;
    std::vector<uint64_t> memsize;
    for (int i = 0; i < mShapeEngine.mDynamicTensors.size(); i++) {
      uint64_t sum = sizeof(float);
      auto tname = mShapeEngine.mDynamicTensors[i].c_str();
      auto n = mShapeEngine.get_tensor_shape_len(tname);
      auto sptr = mShapeEngine.get_tensor_shape_ptr(tname);
      mShapePtr[i] = {n, sptr};
      for (int j = 0; j < n; j++) {
        sum *= sptr[j];
      }
      mTotalMem += sum;
      memsize.push_back(sum);
    }
    mBuffer.resize(mTotalMem, mQueue);
    mPtrs.clear();
    size_t pos = 0;
    for (int i = 0; i < memsize.size(); i++) {
      mPtrs.push_back(mBuffer.data() + pos);
      pos += memsize[i];
    }
  }

  void updateMemory(memory_map_t &_memmap) {
    auto dtensor_count = mShapeEngine.mDynamicTensors.size();
    mPtrs.resize(dtensor_count);
    mShapePtr.resize(dtensor_count);
    uint64_t mTotalMem = static_cast<uint64_t>(_memmap.get_total_size());
    std::vector<uint64_t> memsize;
    for (int i = 0; i < mShapeEngine.mDynamicTensors.size(); i++) {
      uint64_t sum = sizeof(float);
      auto tname = mShapeEngine.mDynamicTensors[i].c_str();
      auto n = mShapeEngine.get_tensor_shape_len(tname);
      auto sptr = mShapeEngine.get_tensor_shape_ptr(tname);
      mShapePtr[i] = {n, sptr};
    }
    if (mTotalMem > mBuffer.size()) {
      mBuffer.resize(mTotalMem, mQueue);
    }
    auto ptr = mBuffer.data();
    for (int i = 0; i < mShapeEngine.mDynamicTensors.size(); i++) {
      auto tname = mShapeEngine.mDynamicTensors[i].c_str();
      mPtrs[i] = mBuffer.data() + _memmap.get_tensor_offset(tname);
    }
  }

  void sync() {
    mQueue.wait();
    mDNNLStream.wait();
    mTimer.record_end();
  }

  void addTensorMemory(const std::vector<int> &indice) {
    uint64_t mTotalMem = 0;
    std::vector<uint64_t> memsize;
    for (int i = 0; i < indice.size(); i++) {
      uint64_t sum = sizeof(float);
      auto tname = mShapeEngine.mDynamicTensors[indice[i]].c_str();
      auto n = mShapeEngine.get_tensor_shape_len(tname);
      auto sptr = mShapeEngine.get_tensor_shape_ptr(tname);
      for (int j = 0; j < n; j++) {
        sum *= sptr[j];
      }
      mTotalMem += sum;
      memsize.push_back(sum);
    }
    mStaticBuffer.resize(mTotalMem, mQueue);
    auto ptr = mStaticBuffer.data();
    for (int i = 0; i < memsize.size(); i++) {
      mPtrs[indice[i]] = ptr;
      ptr += memsize[i];
    }
  }

  shape_engine_t mShapeEngine;
  std::vector<void *> mPtrs;
  std::vector<tensor_shape_t> mShapePtr;
  sycl_vector<char> mStaticBuffer;
  sycl_vector<char> mBuffer;
  std::vector<std::vector<void *>> mLayerInPtr, mLayerOutPtr;
  std::vector<std::vector<tensor_shape_t>> mLayerInShape, mLayerOutShape;
  std::vector<SyclEvent> mEvents;
  TimeRecorder mTimer;
  bool mProfile;
  std::unordered_map<std::string, int> mMaxVariables;
  sycl::queue mQueue;
  dnnl::engine mDNNLEngine;
  dnnl::stream mDNNLStream;
  FckDNNL mFck;
};

class LayerBase {
public:
  LayerBase(const char *_name, attr_map_t &attrs) : mName(_name){};

  virtual ~LayerBase() {}

  virtual sycl::event forward(std::vector<void *> &mInputs,
                              std::vector<tensor_shape_t> &mIShapes,
                              std::vector<void *> &mOutputs,
                              std::vector<tensor_shape_t> &mOShapes,
                              FckDNNL &_stream) = 0;

  virtual void setweights(std::vector<onnx_tool::Tensor> &tensors,
                          sycl::queue &_q) = 0;

  std::string mOp;
  std::string mName;
};

class LayerCreator {
public:
  virtual const char *getType() = 0;
  virtual LayerBase *createLayer(const char *_name, attr_map_t &_attrs) = 0;
};

class LayerFactory {
public:
  static LayerBase *createLayer(const char *_type, const char *_name,
                                attr_map_t &_attrs) {
    for (int i = 0; i < mCreatetors.size(); i++) {
      auto &creator = mCreatetors[i];
      if (strcmp(creator->getType(), _type) == 0) {
        return creator->createLayer(_name, _attrs);
      }
    }
    return nullptr;
  }
  static std::vector<LayerCreator *> mCreatetors;
};

template <typename _T> struct LayerRegistry {
public:
  LayerRegistry(const char *_type) {
    LayerFactory::mCreatetors.push_back(new _T(_type));
  }
};

template <typename _T> class LayerCreatorT : public LayerCreator {
public:
  LayerCreatorT(const char *_type) { mType = _type; }
  LayerBase *createLayer(const char *_name, attr_map_t &_attrs) override {
    auto ptr = new _T(_name, _attrs);
    ptr->mOp = mType;
    return ptr;
  }
  virtual const char *getType() override { return mType.c_str(); }
  std::string mType;
};

#define REGISTER_LAYER(type)                                                   \
  static LayerRegistry<LayerCreatorT<type>> _LayerRegistry##type(#type);

class RuntimeEngine {
public:
  RuntimeEngine() {}

  ~RuntimeEngine() {
    for (int i = 0; i < mLayers.size(); i++) {
      delete mLayers[i];
    }
  }

  void forward(DynamicBindings *_bindings) {
    if (_bindings->mProfile)
      _bindings->mTimer.record_start();
    for (int i = 0; i < mLayers.size(); i++) {
      _bindings->mEvents[i].mEvent = mLayers[i]->forward(
          _bindings->mLayerInPtr[i], _bindings->mLayerInShape[i],
          _bindings->mLayerOutPtr[i], _bindings->mLayerOutShape[i],
          _bindings->mFck);
    }
  }

  void save_proflie(const char *_file, DynamicBindings *_bindings) {
    FILE *fp = fopen(_file, "w");
    fprintf(fp, "Name,Type,TimeMs\n");
    for (int i = 0; i < mLayers.size(); i++) {
      fprintf(fp, "%s,%s,%f\n", mLayers[i]->mName.c_str(),
              mLayers[i]->mOp.c_str(), _bindings->mEvents[i].execute_time());
    }
    fprintf(fp, "Total,_,%f\n", _bindings->mTimer.elapsed_time());
    fclose(fp);
  }
  std::vector<LayerBase *> mLayers;
};

class InferenceContext {
public:
  InferenceContext(onnx_tool::Graph *graph, shape_engine_t &_engine) {
    mGraph.reset(graph);
    mShapeEngine = _engine;
    for (size_t i = 0; i < _engine.mDynamicTensors.size(); i++) {
      mDynamicIndexMap[_engine.mDynamicTensors[i]] = i;
    }
  }

  RuntimeEngine *createRuntimeEngine() {
    auto engine = new RuntimeEngine();
    engine->mLayers.resize(mGraph->mNodes.size());
    auto d_selector{sycl::default_selector_v};
    auto queue = sycl::queue(d_selector, exception_handler);

    for (int i = 0; i < mGraph->mNodes.size(); i++) {
      auto &node = mGraph->mNodes[i];
      auto layerptr = LayerFactory::createLayer(
          node.mOpType.c_str(), node.mName.c_str(), node.mAttributes);
      if (layerptr) {
        std::vector<onnx_tool::Tensor> in_stensors;
        for (int iin = 0; iin < node.mInputNames.size(); iin++) {
          auto name = node.mInputNames[iin];
          auto iter = mDynamicIndexMap.find(name);
          if (iter == mDynamicIndexMap.end()) {
            in_stensors.push_back(mGraph->mTensorMap[name]);
          }
        }
        layerptr->setweights(in_stensors, queue);
      }
      engine->mLayers[i] = layerptr;
    }
    return engine;
  }

  void updateDynamicBindings(DynamicBindings *ptr) {
    auto layercount = mGraph->mNodes.size();
    ptr->mEvents.resize(layercount);
    ptr->mLayerInPtr.resize(layercount);
    ptr->mLayerOutPtr.resize(layercount);
    ptr->mLayerInShape.resize(layercount);
    ptr->mLayerOutShape.resize(layercount);
    for (int i = 0; i < mGraph->mNodes.size(); i++) {
      auto &node = mGraph->mNodes[i];
      auto &in_dptrs = ptr->mLayerInPtr[i];
      auto &out_dptrs = ptr->mLayerOutPtr[i];
      auto &in_shapes = ptr->mLayerInShape[i];
      auto &out_shapes = ptr->mLayerOutShape[i];
      for (int iin = 0; iin < node.mInputNames.size(); iin++) {
        auto name = node.mInputNames[iin];
        auto iter = mDynamicIndexMap.find(name);
        if (iter != mDynamicIndexMap.end()) {
          auto indx = mDynamicIndexMap[name];
          in_dptrs.push_back(ptr->mPtrs[indx]);
          in_shapes.push_back(ptr->mShapePtr[indx]);
        }
      }
      for (int iout = 0; iout < node.mOutputNames.size(); iout++) {
        auto name = node.mOutputNames[iout];
        auto iter = mDynamicIndexMap.find(name);
        if (iter != mDynamicIndexMap.end()) {
          auto indx = mDynamicIndexMap[name];
          out_dptrs.push_back(ptr->mPtrs[indx]);
          out_shapes.push_back(ptr->mShapePtr[indx]);
        }
      }
    }
  }
  DynamicBindings *
  createDynamicBindings(const variable_pairs_t &max_shape, bool profile,
                        const std::vector<std::string> &keep_tnames = {}) {
    auto ptr = new DynamicBindings(mShapeEngine, profile);
    ptr->updateMaxShape(max_shape);
    ptr->updateMemory(max_shape);
    auto indice = tensorName2Index(keep_tnames);
    ptr->addTensorMemory(indice);
    updateDynamicBindings(ptr);
    return ptr;
  }

  DynamicBindings *
  createDynamicBindings(memory_map_t &mem_map, bool profile,
                        const std::vector<std::string> &keep_tnames = {}) {
    auto ptr = new DynamicBindings(mShapeEngine, profile);
    ptr->updateMemory(mem_map);
    ptr->updateMaxShape(mem_map.get_max_shape());
    ptr->reshape(mem_map.get_max_shape());
    auto indice = tensorName2Index(keep_tnames);
    ptr->addTensorMemory(indice);
    updateDynamicBindings(ptr);
    return ptr;
  }

  std::vector<int> tensorName2Index(const std::vector<std::string> &_names) {
    std::vector<int> indice(_names.size());
    for (size_t i = 0; i < _names.size(); i++) {
      indice[i] = mDynamicIndexMap[_names[i]];
    }
    return indice;
  }

  shape_engine_t mShapeEngine;
  std::unordered_map<std::string, int> mDynamicIndexMap;
  std::unique_ptr<onnx_tool::Graph> mGraph;
};
} // namespace sycl_engine
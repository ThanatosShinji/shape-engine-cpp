#include "sycl_engine.h"
using namespace sycl_engine;

void inference_resnet50() {
  const char *filepath = "resnet50_fused.se";
  // Shape engine is very light
  // Each binding should have its unique shape engine
  auto engine = onnx_tool::shape_engine::ShapeEngine();
  engine.deserializeFile(filepath);

  // Avoid to copy this instance
  // Graph contains all weights, you dont want multiple copies
  const char *cfilepath = "resnet50_fused.cg";
  auto ptr = onnx_tool::Graph::deserializeFile(cfilepath);

  // Contruct inference context with compute graph and shape engine
  auto ctx = InferenceContext(ptr, engine);

#if NATIVE_MEMORY
  // Create one dynamic bindings. bindings means a group of dynamic
  // tensors, bindings can be parallelly executed on multiple CPU cores.
  auto dbindings = ctx.createDynamicBindings({{"w", 224}, {"h", 224}}, true);

  // Create another dynamic binginds
  auto dbindings1 = ctx.createDynamicBindings({{"w", 224}, {"h", 224}}, true);
#else
  const char *mfilepath = "resnet50_fused.cm";
  auto memory = onnx_tool::MemoryMap();
  memory.deserializeFile(mfilepath);

  // The MemoryMap is a compressed memory allocator. Using it to create
  // DynamicBindings will save memory space.
  auto dbindings = ctx.createDynamicBindings(memory, true, {"data"});

  // Create another dynamic binginds
  // keep input tensor 'data' to static memory, in case it will be
  // overwritten by inference. MemoryMap is a compressed memory allocator,
  // which may reuse 'data' memory for other tensors.
  auto dbindings1 = ctx.createDynamicBindings(memory, true, {"data"});
#endif

  // Runtime Engine=Op Kernels+static weights
  // one runtime Engine can execute multiple bindings at the same time
  auto runtime = ctx.createRuntimeEngine();

  // Done with createDynamicBindings and createRuntimeEngine, you can release
  // Graph to save memory space.
  ctx.mGraph.reset(nullptr);

  // simple case with profile on
  auto inputidx = ctx.mDynamicIndexMap["data"];   // input tensor
  auto inputptr = dbindings->mPtrs[inputidx];     // input tensor buffer
  auto inputptr1 = dbindings1->mPtrs[inputidx];   // input tensor buffer
  auto in_shape = dbindings->mShapePtr[inputidx]; // input shape pointer
  auto size = std::accumulate(in_shape.ptr, in_shape.ptr + in_shape.n, 1,
                              std::multiplies<int>());
  std::vector<float> hostIn(size, 0.5f);
  dbindings->mQueue.memcpy(inputptr, hostIn.data(), size * 4).wait();
  dbindings->mQueue.memcpy(inputptr1, inputptr, size * 4).wait();

  printf("\n1x3x224x224\n");
  for (int i = 0; i < 10; i++) {
    runtime->forward(dbindings); // inference with this bindings
    dbindings->sync();
  }
  runtime->save_proflie("test.csv", dbindings);
  auto outputidx = ctx.mDynamicIndexMap["resnetv24_dense0_fwd"]; // output
                                                                 // tensor
  auto outputptr = dbindings->mPtrs[outputidx];
  auto out_shape = dbindings->mShapePtr[outputidx]; // output shape pointer
  auto osize = std::accumulate(out_shape.ptr, out_shape.ptr + out_shape.n, 1,
                               std::multiplies<int>());
  std::vector<float> hostOut(osize, 0.f);
  dbindings->mQueue.memcpy(hostOut.data(), outputptr, osize * 4).wait();
  for (int i = 0; i < osize; i++) {
    printf("%f ", hostOut[i]);
  }
  printf("\n");

  // mutiple stream case
  runtime->forward(dbindings);  // inference with this bindings
  runtime->forward(dbindings1); // inference with this bindings
  dbindings->sync();
  dbindings1->sync();
  runtime->save_proflie("test_1.csv", dbindings);
  runtime->save_proflie("test_2.csv", dbindings1);

  // dynamic inference case
  dbindings->reshape({{"h", 112}, {"w", 112}});
  runtime->forward(dbindings);
  dbindings->sync();
  runtime->save_proflie("test112.csv", dbindings);

  delete dbindings;
  delete dbindings1;
  delete runtime;
}

int main() {
  inference_resnet50();
  return 0;
}
#include "cuda_engine.h"
#include <numeric>

using namespace cuda_engine;

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
  // Create one dynamic bindings. bindings means a group of dynamic tensors,
  // multiple bindings instances can be asyncly executed on one GPU .
  auto dbindings =
      ctx.createDynamicBindings({{"w", 224}, {"h", 224}}, true, {"data"});

  // Create another bindings. each bindings has one CUDA stream
  auto dbindings1 =
      ctx.createDynamicBindings({{"w", 224}, {"h", 224}}, true, {"data"});
#else
  const char *mfilepath = "resnet50_fused.cm";
  auto memory = onnx_tool::MemoryMap();
  memory.deserializeFile(mfilepath);

  // The MemoryMap is a compressed memory allocator. Using it to create
  // DynamicBindings will save memory space.
  auto dbindings = ctx.createDynamicBindings(memory, true, {"data"});

  // Create another dynamic binginds
  // keep input tensor 'data' to static memory, in case it will be overwritten
  // by inference. MemoryMap is a compressed memory allocator, which may reuse
  // 'data' memory for other tensors.
  auto dbindings1 = ctx.createDynamicBindings(memory, true, {"data"});
#endif

  // Runtime Engine=Op Kernels+static weights
  // one runtime Engine can execute multiple bindings at the same time
  auto runtime = ctx.createRuntimeEngine();

  // Done with createDynamicBindings and createRuntimeEngine, you can release
  // Graph to save memory space.
  ctx.mGraph.reset(nullptr);

  auto inputidx = ctx.mDynamicIndexMap["data"];          // input tensor
  auto inputptr = (float *)dbindings->mPtrs[inputidx];   // input tensor buffer
  auto inputptr1 = (float *)dbindings1->mPtrs[inputidx]; // input tensor buffer
  auto in_shape = dbindings->mShapePtr[inputidx];        // input shape pointer
  auto size = std::accumulate(in_shape.ptr, in_shape.ptr + in_shape.n, 1,
                              std::multiplies<int>());
  std::vector<float> testinput(size, 0.5f);
  cudaMemcpy(inputptr, testinput.data(), testinput.size() * 4,
             cudaMemcpyHostToDevice);
  cudaMemcpy(inputptr1, testinput.data(), testinput.size() * 4,
             cudaMemcpyHostToDevice);

  printf("\n1x3x224x224\n");
  for (int i = 0; i < 10; i++) {
    runtime->forward(dbindings); // async inference with this bindings
    dbindings->sync();           // CUDA sync
  }

  runtime->save_proflie("test.csv", dbindings);
  auto outputidx = ctx.mDynamicIndexMap["resnetv24_dense0_fwd"]; // output
                                                                 // tensor
  auto outputptr = (float *)dbindings->mPtrs[outputidx];
  auto outputptr1 = (float *)dbindings1->mPtrs[outputidx];
  auto out_shape = dbindings->mShapePtr[outputidx]; // output shape pointer
  auto osize = std::accumulate(out_shape.ptr, out_shape.ptr + out_shape.n, 1,
                               std::multiplies<int>());
  auto testoutput = cuda2Vec(outputptr, osize);
  for (int i = 0; i < osize; i++) {
    printf("%f ", testoutput[i]);
  }
  printf("\n");

  const char *print_layer = "resnetv24_stage1_conv3_fwd";
  for (int i = 0; i < runtime->mLayers.size(); i++) {
    if (runtime->mLayers[i]->mName == print_layer) {
      printf("%s Time:%f\n", print_layer,
             dbindings->mLayerRecorder[i].elapsed_time());
    }
  }
  // async inference two bindings at the same time
  runtime->forward(dbindings);
  runtime->forward(dbindings1);
  dbindings->sync();
  dbindings1->sync();
  runtime->save_proflie("test_1.csv", dbindings);
  runtime->save_proflie("test_2.csv", dbindings1);
  printf("First Binding:\n");
  auto host_b0 = cuda2Vec(outputptr, osize);
  for (int i = 0; i < osize; i++) {
    printf("%f ", host_b0[i]);
  }
  printf("\n");
  printf("Second Binding:\n");
  auto host_b1 = cuda2Vec(outputptr1, osize);
  for (int i = 0; i < osize; i++) {
    printf("%f ", host_b1[i]);
  }
  printf("\n");
  // dynamic input shape inference
  dbindings->reshape({{"h", 112}, {"w", 112}});
  runtime->forward(dbindings);
  dbindings->sync();
  runtime->save_proflie("test112.csv", dbindings);
  delete dbindings;
  delete runtime;
}

int main() {
  inference_resnet50();
  return 0;
}
#include "simple_inference_engine.h"
#include <algorithm>
#include <numeric>
#include <thread>

using namespace simple_inference_engine_f32;

void topk(const float *src, size_t size, const int k, const float filtervalue,
          float *dst, float *maxvalue) {
  std::memcpy(dst, src, size * sizeof(float));
  std::sort(dst, dst + size);
  auto threshold = dst[size - k];
  auto maxv = filtervalue;
  for (size_t i = 0; i < size; i++) {
    auto val = src[i] <= threshold ? filtervalue : src[i];
    dst[i] = val;
    maxv = val > maxv ? val : maxv;
  }
  *maxvalue = maxv;
}

void softmax(const float *src, size_t size, float maxvalue, float *prob) {
  double expsum = 0;
  for (size_t i = 0; i < size; i++) {
    prob[i] = expf(src[i] - maxvalue);
    expsum += prob[i];
  }
  float scale = float(1.0 / expsum);
  for (size_t i = 0; i < size; i++) {
    prob[i] *= scale;
  }
}
#include <random>
size_t prob_sample(const float *prob, size_t size) {
  auto rv = float(std::rand()) / RAND_MAX;
  float accv = 0.f;
  for (size_t i = 0; i < size; i++) {
    if (accv <= rv && accv + prob[i] > rv) {
      return i;
    }
    accv += prob[i];
  }
  return size - 1;
}

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
  // bindings can be parallelly executed on multiple CPU cores.
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

  // simple case with profile on
  auto inputidx = ctx.mDynamicIndexMap["data"];          // input tensor
  auto inputptr = (float *)dbindings->mPtrs[inputidx];   // input tensor buffer
  auto inputptr1 = (float *)dbindings1->mPtrs[inputidx]; // input tensor buffer
  auto in_shape = dbindings->mShapePtr[inputidx];        // input shape pointer
  auto size = std::accumulate(in_shape.ptr, in_shape.ptr + in_shape.n, 1,
                              std::multiplies<int>());
  for (int i = 0; i < size; i++) {
    inputptr[i] = 0.5f;
    inputptr1[i] = 0.5f;
  }

  printf("\n1x3x224x224\n");
  for (int i = 0; i < 1; i++) {
    runtime->forward(dbindings); // inference with this bindings
  }
  runtime->save_proflie("test.csv", dbindings);

  auto outputidx = ctx.mDynamicIndexMap["resnetv24_dense0_fwd"]; // output
                                                                 // tensor
  auto outputptr = (float *)dbindings->mPtrs[outputidx];
  auto out_shape = dbindings->mShapePtr[outputidx]; // output shape pointer
  auto osize = std::accumulate(out_shape.ptr, out_shape.ptr + out_shape.n, 1,
                               std::multiplies<int>());
  for (int i = 0; i < osize; i++) {
    printf("%f ", outputptr[i]);
  }
  printf("\n");

  // multiple threads case
  int constexpr ThreadsCount = 2;
  std::thread threads[ThreadsCount];
  DynamicBindings *db_ptrs[2] = {dbindings, dbindings1};
  for (int i = 0; i < ThreadsCount; i++) {
    threads[i] = std::thread(
        [runtime](DynamicBindings *bind) { runtime->forward(bind); },
        db_ptrs[i]);
  }
  for (int i = 0; i < ThreadsCount; i++) {
    threads[i].join();
    auto tmpptr = (float *)db_ptrs[i]->mPtrs[outputidx];
    printf("Thread %d Result:\n", i);
    for (int ii = 0; ii < osize; ii++) {
      printf("%f ", tmpptr[ii]);
    }
    printf("\n");
  }
  runtime->save_proflie("test_1.csv", dbindings);
  runtime->save_proflie("test_2.csv", dbindings1);

  // dynamic inference case
  dbindings->reshape({{"h", 112}, {"w", 112}});
  runtime->forward(dbindings);
  runtime->save_proflie("test112.csv", dbindings);

  delete dbindings;
  delete dbindings1;
  delete runtime;
}

void inference_gpt2() {
  const char *filepath = "gpt2.se";
  // Shape engine is very light
  // Each binding should have its unique shape engine
  auto engine = onnx_tool::shape_engine::ShapeEngine();
  engine.deserializeFile(filepath);

  // Avoid to copy this instance
  // Graph contains all weights, you dont want multiple copies
  const char *cfilepath = "gpt2.cg";
  auto ptr = onnx_tool::Graph::deserializeFile(cfilepath);

  // Contruct inference context with compute graph and shape engine
  auto ctx = InferenceContext(ptr, engine);

  const char *mfilepath = "gpt2.cm";
  auto memory = onnx_tool::MemoryMap();
  memory.deserializeFile(mfilepath);

  // The MemoryMap is a compressed memory allocator. Using it to create
  // DynamicBindings will save memory space.
  const char *debugtensor = "238";
  auto dbindings =
      ctx.createDynamicBindings(memory, false, {debugtensor, "input1"});

  // Runtime Engine=Op Kernels+static weights
  // one runtime Engine can execute multiple bindings at the same time
  auto runtime = ctx.createRuntimeEngine();

  // Done with createDynamicBindings and createRuntimeEngine, you can release
  // Graph to save memory space.
  ctx.mGraph.reset(nullptr);

  int testprompt[] = {15496, 11, 314, 1101, 257, 3303, 2746, 11};
  int ninput = sizeof(testprompt) / sizeof(testprompt[0]);
  dbindings->reshape({{"seq", ninput}});
  int ngen = 10;
  // simple case with profile on
  auto inputidx = ctx.mDynamicIndexMap["input1"];    // input tensor
  auto inputptr = (int *)dbindings->mPtrs[inputidx]; // input tensor buffer
  auto in_shape = dbindings->mShapePtr[inputidx];    // input shape pointer
  auto size = std::accumulate(in_shape.ptr, in_shape.ptr + in_shape.n, 1,
                              std::multiplies<int>());
  auto outputidx = ctx.mDynamicIndexMap["output1"];
  auto outputptr = (float *)dbindings->mPtrs[outputidx];
  auto debugptr = (float *)dbindings->mPtrs[ctx.mDynamicIndexMap[debugtensor]];
  auto out_shape = dbindings->mShapePtr[outputidx]; // output shape pointer
  int lm_size = out_shape.ptr[3];
  std::vector<float> topkout(lm_size), prob(lm_size);

  for (int i = 0; i < size; i++) {
    inputptr[i] = testprompt[i];
  }
  float valmax = 0.f;
  srand(42);
  for (size_t i = 0; i < ngen; i++) {
    dbindings->reshape({{"seq", ninput + i}});
    runtime->forward(dbindings); // inference with this bindings
    topk(outputptr + (ninput + i - 1) * lm_size, lm_size, 50, -10000.f,
         topkout.data(), &valmax);
    softmax(topkout.data(), lm_size, valmax, prob.data());
    auto nextids = prob_sample(prob.data(), lm_size);
    inputptr[ninput + i] = nextids;
    printf("%d,", nextids);
  }

  // runtime->save_proflie("test.csv", dbindings);
  printf("\n");
}

int main() {
  // inference_resnet50();
  inference_gpt2();
  return 0;
}
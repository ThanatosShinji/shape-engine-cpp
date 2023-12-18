#include "x64_inference_engine.h"
#include <algorithm>
#include <numeric>
#include <thread>

using namespace x64_inference_engine_f32;

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

void inference_gpt2_kvcache() {
  const char *filepath = "gpt2_kvcache.se";
  // Shape engine is very light
  // Each binding should have its unique shape engine
  auto engine = onnx_tool::shape_engine::ShapeEngine();
  engine.deserializeFile(filepath);

  // Avoid to copy this instance
  // Graph contains all weights, you dont want multiple copies
  const char *cfilepath = "gpt2_kvcache.cg";
  auto ptr = onnx_tool::Graph::deserializeFile(cfilepath);

  // Contruct inference context with compute graph and shape engine
  auto ctx = InferenceContext(ptr, engine);

  const char *mfilepath = "gpt2_kvcache.cm";
  auto memory = onnx_tool::MemoryMap();
  memory.deserializeFile(mfilepath);

  // The MemoryMap is a compressed memory allocator. Using it to create
  // DynamicBindings will save memory space.
  const char *debugtensor = "238";
  std::vector<std::string> kvtensors(12);
  for (size_t i = 0; i < 12; i++) {
    kvtensors[i] = std::string("output") + std::to_string(2 + i);
  }
  auto dbindings = ctx.createDynamicBindings(memory, true, kvtensors);

  // Runtime Engine=Op Kernels+static weights
  // one runtime Engine can execute multiple bindings at the same time
  auto runtime = ctx.createRuntimeEngine();
  runtime->set_threads(16);
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
  auto npast_ptr = (int64_t *)dbindings->mPtrs[ctx.mDynamicIndexMap["n_past"]];
  auto in_shape = dbindings->mShapePtr[inputidx]; // input shape pointer
  auto size = std::accumulate(in_shape.ptr, in_shape.ptr + in_shape.n, 1,
                              std::multiplies<int>());
  auto outputidx = ctx.mDynamicIndexMap["output1"];
  auto outputptr = (float *)dbindings->mPtrs[outputidx];
  auto debugptr = (float *)dbindings->mPtrs[ctx.mDynamicIndexMap[debugtensor]];
  auto out_shape = dbindings->mShapePtr[outputidx]; // output shape pointer
  int lm_size = out_shape.ptr[3];
  std::vector<float> topkout(lm_size), prob(lm_size);
  float valmax = 0.f;
  srand(42);

  for (int i = 0; i < size; i++) {
    inputptr[i] = testprompt[i];
  }
  *npast_ptr = 0;
  //warm up
  dbindings->reshape({{"seq", ninput}});
  runtime->forward(dbindings);
#if 1
  dbindings->reshape({{"seq", ninput}});
  runtime->forward(dbindings);
  topk(outputptr + (ninput - 1) * lm_size, lm_size, 50, -10000.f,
       topkout.data(), &valmax);
  softmax(topkout.data(), lm_size, valmax, prob.data());
  auto nextids = prob_sample(prob.data(), lm_size);
  printf("%d,", nextids);
  dbindings->reshape({{"seq", 1}});
  auto firstProf = dbindings->mLayerRecorder;
  auto firsttotal = dbindings->mTotalRecorder;
  *npast_ptr = ninput;
  for (size_t i = 1; i < ngen; i++) {
    inputptr[0] = nextids;
    runtime->forward(dbindings);
    topk(outputptr, lm_size, 50, -10000.f, topkout.data(), &valmax);
    softmax(topkout.data(), lm_size, valmax, prob.data());
    nextids = prob_sample(prob.data(), lm_size);
    (*npast_ptr)++;
    printf("%d,", nextids);
  }
#else
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

#endif
  runtime->save_proflie("test_next.csv", dbindings);
  dbindings->mLayerRecorder = firstProf;
  dbindings->mTotalRecorder = firsttotal;
  runtime->save_proflie("test_first.csv", dbindings);
  printf("\n");
}

int main() {
  inference_gpt2_kvcache();
  return 0;
}
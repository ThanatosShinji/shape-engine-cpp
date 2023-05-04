#include "cuda_engine.h"
#include <numeric>

using namespace cuda_engine;

void inference_resnet50()
{
	const char* filepath = "resnet50_fused.se";
	//Shape engine is very light
	//Each binding should have its unique shape engine
	auto engine = onnx_tool::shape_engine::ShapeEngine();
	engine.deserializeFile(filepath);

	//Avoid to copy this instance
	//Graph contains all weights, you dont want multiple copies
	const char* cfilepath = "resnet50_fused.cg";
	auto ptr = onnx_tool::Graph::deserializeFile(cfilepath);

	//Contruct inference context with compute graph and shape engine
	auto ctx = InferenceContext(ptr, engine);

	//Create one dynamic bindings. bindings means a group of dynamic tensors,
	//bindings can be parallelly executed on multiple CPU cores.
	auto dbindings = ctx.createDynamicBindings({ {"w",224},{"h",224} });

	//Runtime Engine=Op Kernels+static weights
	//one runtime Engine can only execute one bindings at the same time
	auto runtime = ctx.createRuntimeEngine();

	//Done with createDynamicBindings and createRuntimeEngine, you can release Graph to save memory space.
	ctx.mGraph.reset(nullptr);

	auto inputidx = ctx.mDynamicIndexMap["data"];//input tensor
	auto inputptr = (float*)dbindings->mPtrs[inputidx];//input tensor buffer
	auto in_shape = dbindings->mShapePtr[inputidx];//input shape pointer
	auto size = std::accumulate(in_shape.ptr, in_shape.ptr + in_shape.n, 1, std::multiplies<int>());
	std::vector<float> testinput(size, 0.5f);
	cudaMemcpy(inputptr, testinput.data(), testinput.size() * 4, cudaMemcpyHostToDevice);

	printf("\n1x3x224x224\n");
	runtime->mProfile = true;
	for (int i = 0; i < 10; i++)
	{
		runtime->forward(dbindings);//inference with this bindings
		dbindings->sync();
	}

	runtime->save_proflie("test.csv");
	auto outputidx = ctx.mDynamicIndexMap["resnetv24_dense0_fwd"];//output tensor
	auto outputptr = (float*)dbindings->mPtrs[outputidx];
	auto out_shape = dbindings->mShapePtr[outputidx];//output shape pointer
	auto osize = std::accumulate(out_shape.ptr, out_shape.ptr + out_shape.n, 1, std::multiplies<int>());
	auto testoutput = cuda2Vec(outputptr, osize);
	for (int i = 0; i < osize; i++)
	{
		printf("%f ", testoutput[i]);
	}
	printf("\n");

	const char* print_layer = "resnetv24_stage1_conv3_fwd";
	for (int i = 0; i < runtime->mLayers.size(); i++)
	{
		if (runtime->mLayers[i]->mName == print_layer)
		{
			printf("%s Time:%f\n", print_layer, runtime->mLayers[i]->mRecoder.elapsed_time());
		}
	}

	dbindings->reshape({ {"h",384 }, { "w",384 } });
	runtime->forward(dbindings);
	dbindings->sync();
	runtime->save_proflie("test384.csv");
	delete dbindings;
	delete runtime;
}

int main()
{
	inference_resnet50();
	return 0;
}
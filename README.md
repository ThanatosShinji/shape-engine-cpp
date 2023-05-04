<a href="README_CN.md">简体中文</a>
# `shape-engine-cpp`

A tiny inference sample with onnx-tool's compute graph and shape engine.  It implements cpp version of onnx-tool's compute graph and shape engine loader.   
This repo also creates a tiny inference engine template for ResNet18 model. The size of compiled engine is 76KB.

  
## Compute Graph & Shape Engine  
---
These two binary files are created by onnx-tool, please refer [link](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/inference_engine.md).  
"resnet18.cg" is the serialized compute graph of ResNet18. "resnet18.se" is the serialized shape engine of ResNet18. 
~~~cpp
	const char* filepath = "resnet18.se";
	//Shape engine is very light
	//Each binding should have its unique shape engine
	auto engine = onnx_tool::shape_engine::ShapeEngine();
	engine.deserializeFile(filepath);

	//Avoid to copy this instance
	//Graph contains all weights, you dont want multiple copies
	const char* cfilepath = "resnet18.cg";
	auto ptr = onnx_tool::Graph::deserializeFile(cfilepath);
~~~

## Tiny Inference Engine
---
The inference engine is based on onnx_tool::shape_engine::ShapeEngine and onnx_tool::Graph instances.
~~~cpp
    //Contruct inference context with compute graph and shape engine
	auto ctx = InferenceContext(ptr, engine);

	//Create one dynamic bindings. bindings means a group of dynamic tensors,
	//bindings can be parallelly executed on multiple CPU cores.
	auto dbindings = ctx.createDynamicBindings();
	auto maxshape = variable_pairs_t{ {"w",224},{"h",224} };
	dbindings->updateMemory(maxshape);//memory allocation for Maxmimum input shapes

	//Runtime Engine=Op Kernels+static weights
	//one runtime Engine can only execute one bindings at the same time
	auto runtime = ctx.createRuntimeEngine();
~~~

## Inference with dynamic input shapes
---
Fill input tensor with float value of 0.5f
~~~cpp
	auto inputidx=ctx.mDynamicIndexMap["data"];//input tensor
	auto inputptr = (float*)dbindings->mPtrs[inputidx];//input tensor buffer
	auto in_shape = dbindings->mShapePtr[inputidx];//input shape pointer
	auto size = std::accumulate(in_shape.ptr, in_shape.ptr + in_shape.n, 1,std::multiplies<int>());
	for (int i = 0; i < size; i++)
	{
		inputptr[i] = 0.5f;
	}
~~~
Do one init inference, bind the dbindings to runtime engine
~~~cpp
	printf("\n1x3x224x224\n");
	runtime->forward_dynamic(dbindings);//inference with this bindings
~~~
Reshape the input shape to 1x3x122x112, and do the second inference.
~~~cpp
    auto newshape = variable_pairs_t{ {"w",112},{"h",122} };
	dbindings->reshape(newshape);//all shapes in this bindings will be updated, less than 1us.
	printf("\n1x3x112x122\n");
	runtime->forward();//reuse last dynamic bindings
~~~
Print output tensor values
~~~cpp
    auto outputidx = ctx.mDynamicIndexMap["resnetv15_dense0_fwd"];//output tensor
	auto outputptr = (float*)dbindings->mPtrs[outputidx];
	auto out_shape = dbindings->mShapePtr[outputidx];//output shape pointer
	auto osize = std::accumulate(out_shape.ptr, out_shape.ptr + out_shape.n, 1, std::multiplies<int>());
	for (int i = 0; i < osize; i++)
	{
		printf("%f ", outputptr[i]);
	}
	printf("\n");
~~~
The whole cpp inference engine for ResNet18 is compiled as a 76KB binary program. You can feed this engine with input tensor of any resolution without much overhead.

## Build and Run
---
- Download [ResNet18](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet18-v1-7.onnx)
- Use onnx-tool to get serialized compute graph and shape engine. [link](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/inference_engine.md)
- Compile and run on Linux:
~~~bash
git clone https://github.com/ThanatosShinji/shape-engine-cpp.git
cd shape-engine-cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cp <path of serializaion>/resnet18.cg ./simple_inference_engine/
cp <path of serializaion>/resnet18.se ./simple_inference_engine/
./simple_inference_engine/simple_inference_engine
~~~
- Windows:  
1. open shape-engine-cpp with VisualStudio(greater than 2019).   
2. Build the project with CMake configure. 
3. Copy the serialization files to simple_inference_engine.exe's path(like: shape-engine-cpp\out\build\x64-mini\simple_inference_engine).  
4. Run simple_inference_engine.exe.
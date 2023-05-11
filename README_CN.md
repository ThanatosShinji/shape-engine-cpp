# `shape-engine-cpp`

本项目实现了onnx-tool中的compute graph和shape engine的cpp类. 可以让cpp代码加载通过onnx-tool序列化的compute graph和shape engine.
可以实现高效率的动态输入更新全图的tensor shapes.另外本项目实现了一个cpp-based推理引擎, 一个cuda-based推理引擎, 实现了compute graph和shape engine的集成调用.
由onnx-tool实现了大多数的graph优化和融合逻辑, 推理引擎只专注于计算方法的实现.

Project | Language | Platform Support | Device | Library 
---|---|---|---
simple_inference_engine | cpp |  All | All | None
simple_cuda_engine | CUDA cpp | Windows Linux | Nvidia GPUs | cuBlas cuDNN
simple_sycl_engine | SYCL cpp | Windows Linux | OpenCL Devices | oneMKL oneDNN
  
## Compute Graph & Shape Engine  
---
Compute Graph代表了计算的真实图, 只包含了计算层, onnx-tool已经移除了tensor shape的转换层(比如ResNet50的flatten), 导出方式参考[link](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/inference_engine.md).  
"resnet50.cg" 是序列化的ResetNet50的compute graph. "resnet50.se" 是序列化的ResNet50的shape engine. 
~~~cpp
const char* filepath = "resnet50_fused.se";
//shape engine可以通过少量的shape计算更新所有tensor的shape
//几乎不占内存, 可以任意拷贝
auto engine = onnx_tool::shape_engine::ShapeEngine();
engine.deserializeFile(filepath);

//graph中包含了图结构和权重, 所以会比较大, 不要拷贝这个Graph, 全局只需要保存一份
const char* cfilepath = "resnet50_fused.cg";
auto ptr = onnx_tool::Graph::deserializeFile(cfilepath);
~~~

## Tiny Inference Engine
---  

~~~cpp
//通过Compute Graph和Shape Engine构造一个推理的Context
auto ctx = InferenceContext(ptr, engine);

//通过Context创建DynamicBindings, 即一个保存Graph临时tensor的对象.(包含所有动态tensor的内存或者显存)
//w和h是shape engine中定义的输入tensor的动态维度(1,3,h,w), 这里通过h=w=224创建所有动态tensor的内存.
auto dbindings = ctx.createDynamicBindings({ {"w",224},{"h",224} });

//Runtime Engine是推理中唯一需要的对象, 包含了每个op的实现算子和各自缓存的权重.
//Runtime Engine可以在同一时间同时对多块DynamicBindings进行推理,是线程安全的, 可以共享推理的权重数据.
auto runtime = ctx.createRuntimeEngine();

//权重已经解析到了Runtime Engine中, Context中的原始Graph已经没有作用, 这里释放掉节省内存
ctx.mGraph.reset(nullptr);
~~~

## Inference with dynamic input shapes
---
为输入tensor("data")全部填入0.5f的初始值.
~~~cpp
auto inputidx = ctx.mDynamicIndexMap["data"];//input tensor
auto inputptr = (float*)dbindings->mPtrs[inputidx];//input tensor buffer
auto in_shape = dbindings->mShapePtr[inputidx];//input shape pointer
auto size = std::accumulate(in_shape.ptr, in_shape.ptr + in_shape.n, 1, std::multiplies<int>());
for (int i = 0; i < size; i++)
{
	inputptr[i] = 0.5f;
}
~~~
对Runtime Engine完成10次推理, 并且打开Profile记录每一层的推理耗时. 并把profiling结果导出到"test.csv"中.
~~~cpp
runtime->mProfile = true;
for (int i = 0; i < 10; i++)
{
	runtime->forward(dbindings);//inference with this bindings
	dbindings->sync(); //CUDA 同步
}
runtime->save_proflie("test.csv");
~~~
更改输入tensor的分辨率为128x128再次推理
~~~cpp
dbindings->reshape({ {"h",128 }, { "w",128 } });//更新输入图像为128x128的分辨率
runtime->forward(dbindings);//推理新的分辨率图像
dbindings->sync(); //CUDA 同步
~~~
查看推理结果, 与onnxruntime离线对比.
~~~cpp
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
~~~


## Build and Run
---
- Download [ResNet50](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v2-7.onnx)
- Use onnx-tool to get serialized compute graph and shape engine. [link](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/inference_engine.md)
- Compile and run on Linux:
~~~bash
git clone https://github.com/ThanatosShinji/shape-engine-cpp.git
cd shape-engine-cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cp <path of serializaion>/resnet50.cg ./simple_inference_engine/
cp <path of serializaion>/resnet50.se ./simple_inference_engine/
./simple_inference_engine/simple_inference_engine
~~~
- Windows:  
1. open shape-engine-cpp with VisualStudio(greater than 2019).   
2. Build the project with CMake configure. 
3. Copy the serialization files to simple_inference_engine.exe's path(like: shape-engine-cpp\out\build\x64-mini\simple_inference_engine).  
4. Run simple_inference_engine.exe.
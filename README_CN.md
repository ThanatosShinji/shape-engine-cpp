# `shape-engine-cpp`

����Ŀʵ����onnx-tool�е�compute graph��shape engine��cpp��. ������cpp�������ͨ��onnx-tool���л���compute graph��shape engine.
����ʵ�ָ�Ч�ʵĶ�̬�������ȫͼ��tensor shapes.���Ȿ��Ŀʵ����һ��cpp-based��������, һ��cuda-based��������, ʵ����compute graph��shape engine�ļ��ɵ���.
��onnx-toolʵ���˴������graph�Ż����ں��߼�, ��������ֻרע�ڼ��㷽����ʵ��.

Project | Language | Platform Support | Device | Library 
---|---|---|---
simple_inference_engine | cpp |  All | All | None
simple_cuda_engine | CUDA cpp | Windows Linux | Nvidia GPUs | cuBlas cuDNN
simple_sycl_engine | SYCL cpp | Windows Linux | OpenCL Devices | oneMKL oneDNN
  
## Compute Graph & Shape Engine  
---
Compute Graph�����˼������ʵͼ, ֻ�����˼����, onnx-tool�Ѿ��Ƴ���tensor shape��ת����(����ResNet50��flatten), ������ʽ�ο�[link](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/inference_engine.md).  
"resnet50.cg" �����л���ResetNet50��compute graph. "resnet50.se" �����л���ResNet50��shape engine. 
~~~cpp
const char* filepath = "resnet50_fused.se";
//shape engine����ͨ��������shape�����������tensor��shape
//������ռ�ڴ�, �������⿽��
auto engine = onnx_tool::shape_engine::ShapeEngine();
engine.deserializeFile(filepath);

//graph�а�����ͼ�ṹ��Ȩ��, ���Ի�Ƚϴ�, ��Ҫ�������Graph, ȫ��ֻ��Ҫ����һ��
const char* cfilepath = "resnet50_fused.cg";
auto ptr = onnx_tool::Graph::deserializeFile(cfilepath);
~~~

## Tiny Inference Engine
---  

~~~cpp
//ͨ��Compute Graph��Shape Engine����һ�������Context
auto ctx = InferenceContext(ptr, engine);

//ͨ��Context����DynamicBindings, ��һ������Graph��ʱtensor�Ķ���.(�������ж�̬tensor���ڴ�����Դ�)
//w��h��shape engine�ж��������tensor�Ķ�̬ά��(1,3,h,w), ����ͨ��h=w=224�������ж�̬tensor���ڴ�.
auto dbindings = ctx.createDynamicBindings({ {"w",224},{"h",224} });

//Runtime Engine��������Ψһ��Ҫ�Ķ���, ������ÿ��op��ʵ�����Ӻ͸��Ի����Ȩ��.
//Runtime Engine������ͬһʱ��ͬʱ�Զ��DynamicBindings��������,���̰߳�ȫ��, ���Թ��������Ȩ������.
auto runtime = ctx.createRuntimeEngine();

//Ȩ���Ѿ���������Runtime Engine��, Context�е�ԭʼGraph�Ѿ�û������, �����ͷŵ���ʡ�ڴ�
ctx.mGraph.reset(nullptr);
~~~

## Inference with dynamic input shapes
---
Ϊ����tensor("data")ȫ������0.5f�ĳ�ʼֵ.
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
��Runtime Engine���10������, ���Ҵ�Profile��¼ÿһ��������ʱ. ����profiling���������"test.csv"��.
~~~cpp
runtime->mProfile = true;
for (int i = 0; i < 10; i++)
{
	runtime->forward(dbindings);//inference with this bindings
	dbindings->sync(); //CUDA ͬ��
}
runtime->save_proflie("test.csv");
~~~
��������tensor�ķֱ���Ϊ128x128�ٴ�����
~~~cpp
dbindings->reshape({ {"h",128 }, { "w",128 } });//��������ͼ��Ϊ128x128�ķֱ���
runtime->forward(dbindings);//�����µķֱ���ͼ��
dbindings->sync(); //CUDA ͬ��
~~~
�鿴������, ��onnxruntime���߶Ա�.
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
#include <cstdio>
#include <cstdlib>
#include <GL/_OpenGL.h>
#include <GL/_Window.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <OptiX/_OptiX_7.h>
#include "Define.h"
#include <_Time.h>
#include <_STL.h>

void initRandom(curandState* state, int seed, unsigned int block, unsigned int grid, unsigned int MaxNum);

#define ElemIndex(rec, index) ( (&(rec->position.x))[index] )

template<class Elem> __inline void swap(Elem* list, int index1, int index2)
{
	Elem temp = list[index1];
	list[index1] = list[index2];
	list[index2] = temp;
}

template<class Elem> int partition(Elem* list, int axis, int left, int right, int pivotIndex)
{
	Elem pivotValue = list[pivotIndex];
	swap(list, right, pivotIndex);
	pivotIndex = right;
	left--;
	while (true)
	{
		do { left++; } while (left < right && ElemIndex(list[left], axis) < ElemIndex(pivotValue, axis));
		do { right--; } while (left < right && ElemIndex(list[right], axis) > ElemIndex(pivotValue, axis));
		if (left < right)
			swap(list, left, right);
		else {
			// Put the pivotValue back in place
			swap(list, left, pivotIndex);
			return left;
		}
	}
}

// find the k-th largest Elem in list, k start from 0
template<class Elem> Elem select(Elem* list, int axis, int left, int right, int k)
{
	while (true)
	{
		// select a value to pivot around between left and right and store the index to it.
		int pivotIndex = (left + right) / 2;
		// Determine where this value ended up.
		int pivotNewIndex = partition<Elem>(list, axis, left, right, pivotIndex);
		if (k == pivotNewIndex) {
			// We found the kth value
			return list[k];
		}
		else if (k < pivotNewIndex)
			// if instead we found the k+Nth value, remove the segment of the list
			// from pivotNewIndex onward from the search.
			right = pivotNewIndex - 1;
		else
			// We found the k-Nth value, remove the segment of the list from
			// pivotNewIndex and below from the search.
			left = pivotNewIndex + 1;
	}
}

namespace CUDA
{
	namespace OptiX
	{
		struct PathTracing :RayTracer
		{
			Context context;
			OptixModuleCompileOptions rt_moduleCompileOptions;
			OptixModuleCompileOptions pt_moduleCompileOptions;
			OptixPipelineCompileOptions rt_pipelineCompileOptions;
			OptixPipelineCompileOptions pt_pipelineCompileOptions;
			ModuleManager mm;
			OptixProgramGroupOptions rt_programGroupOptions;
			OptixProgramGroupOptions pt_programGroupOptions;
			Program rt_raygen;
			Program rt_hitRayRadiance;
			Program rt_hitShadowRay;
			Program rt_missRayRadiance;
			Program rt_missShadowRay;
			Program pt_raygen;
			Program pt_closestHit;
			Program pt_miss;
			OptixPipelineLinkOptions rt_pipelineLinkOptions;
			OptixPipelineLinkOptions pt_pipelineLinkOptions;
			Pipeline rt_pip;
			Pipeline pt_pip;
			SbtRecord<Rt_RayGenData> rt_raygenData;
			SbtRecord<Rt_HitData> rt_hitDatas[RayCount];
			SbtRecord<Pt_RayGenData> pt_raygenData;
			SbtRecord<Pt_HitData> pt_hitData;
			LightSource lightSource;
			Buffer lightSourceBuffer;
			Buffer photonBuffer;
			Buffer photonMapBuffer;
			Buffer rt_raygenDataBuffer;
			Buffer rt_hitDataBuffer;
			Buffer rt_missDataBuffer;
			Buffer pt_raygenDataBuffer;
			Buffer pt_hitDataBuffer;
			Buffer pt_missDataBuffer;
			OptixShaderBindingTable rt_sbt;
			OptixShaderBindingTable pt_sbt;
			Buffer frameBuffer;
			CUstream cuStream;
			Parameters paras;
			Buffer parasBuffer;
			STL box;
			Buffer vertices;
			Buffer normals;
			Buffer kds;
			Buffer debugDatas;
			OptixBuildInput triangleBuildInput;
			OptixAccelBuildOptions accelOptions;
			Buffer GASOutput;
			OptixTraversableHandle GASHandle;
			bool photonFlag;
			PathTracing(OpenGL::SourceManager* _sourceManager, OpenGL::OptiXDefautRenderer* dr, OpenGL::FrameScale const& _size, void* transInfoDevice)
				:
				context(),
				rt_moduleCompileOptions{
					OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
					OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
					OPTIX_COMPILE_DEBUG_LEVEL_NONE },
				pt_moduleCompileOptions{
					OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
					OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
					OPTIX_COMPILE_DEBUG_LEVEL_NONE },
				rt_pipelineCompileOptions{ false,
					OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
					8,2,OPTIX_EXCEPTION_FLAG_NONE,"paras", unsigned int(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE) },//OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE: new in OptiX7.1.0
				pt_pipelineCompileOptions{ false,
					OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
					8,2,OPTIX_EXCEPTION_FLAG_NONE,"paras", unsigned int(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE) },
				mm(&_sourceManager->folder, context, &rt_moduleCompileOptions, &rt_pipelineCompileOptions),
				rt_programGroupOptions{},
				pt_programGroupOptions{},
				pt_miss(Vector<String<char>>("__miss__Ahh"), Program::Miss, &rt_programGroupOptions, context, &mm),
				rt_raygen(Vector<String<char>>("__raygen__RayAllocator"), Program::RayGen, &rt_programGroupOptions, context, &mm),
				rt_hitRayRadiance(Vector<String<char>>("__closesthit__RayRadiance"), Program::HitGroup, &rt_programGroupOptions, context, &mm),
				rt_hitShadowRay(Vector<String<char>>("__closesthit__ShadowRay"), Program::HitGroup, &rt_programGroupOptions, context, &mm),
				rt_missRayRadiance(Vector<String<char>>("__miss__RayRadiance"), Program::Miss, &rt_programGroupOptions, context, &mm),
				rt_missShadowRay(Vector<String<char>>("__miss__ShadowRay"), Program::Miss, &rt_programGroupOptions, context, &mm),
				pt_raygen(Vector<String<char>>("__raygen__PhotonEmit"), Program::RayGen, &pt_programGroupOptions, context, &mm),
				pt_closestHit(Vector<String<char>>("__closesthit__PhotonHit"), Program::HitGroup, &pt_programGroupOptions, context, &mm),
				rt_pipelineLinkOptions{ 1,OPTIX_COMPILE_DEBUG_LEVEL_NONE },//no overrideUsesMotionBlur in OptiX7.1.0
				pt_pipelineLinkOptions{ 10,OPTIX_COMPILE_DEBUG_LEVEL_NONE }, // NOTE: maxDepth = 10 in photon trace stage
				rt_pip(context, &rt_pipelineCompileOptions, &rt_pipelineLinkOptions, { rt_raygen ,rt_hitRayRadiance, rt_hitShadowRay, rt_missRayRadiance, rt_missShadowRay }),
				pt_pip(context, &pt_pipelineCompileOptions, &pt_pipelineLinkOptions, { pt_raygen ,pt_closestHit, pt_miss }),
				lightSourceBuffer(lightSource, false),
				photonBuffer(Buffer::Device),
				photonMapBuffer(Buffer::Device),
				pt_missDataBuffer(Buffer::Device),
				rt_raygenDataBuffer(rt_raygenData, false),
				rt_hitDataBuffer(Buffer::Device),
				rt_missDataBuffer(Buffer::Device),
				pt_raygenDataBuffer(pt_raygenData, false),
				pt_hitDataBuffer(pt_hitData, false),
				rt_sbt({}),
				pt_sbt({}),
				frameBuffer(*dr),
				parasBuffer(paras, false),
				box(_sourceManager->folder.find("resources/boxnew.stl").readSTL()),
				vertices(Buffer::Device),
				normals(Buffer::Device),
				kds(Buffer::Device),
				debugDatas(Buffer::Device),
				triangleBuildInput({}),
				accelOptions({}),
				GASOutput(Buffer::Device),
				GASHandle(0),
				photonFlag(true)
			{
				box.getVerticesRepeated();
				box.getNormals();
				box.printInfo(false);
				vertices.copy(box.verticesRepeated.data, sizeof(Math::vec3<float>)* box.verticesRepeated.length);
				normals.copy(box.normals.data, sizeof(Math::vec3<float>)* box.normals.length);

				float3* kdsTemp = new float3[box.normals.length];
				for (int c0(0); c0 < box.normals.length; c0++)
					kdsTemp[c0] = { 0.73f, 0.73f, 0.73f };
				kdsTemp[20] = make_float3(0.65f, 0.05f, 0.05f);
				kdsTemp[21] = make_float3(0.65f, 0.05f, 0.05f);
				kdsTemp[24] = make_float3(0.12f, 0.45f, 0.15f);
				kdsTemp[25] = make_float3(0.12f, 0.45f, 0.15f);
				kds.copy(kdsTemp, sizeof(float3)* box.normals.length);
				delete[] kdsTemp;

				lightSource.position = { 0.0f, 0.99f, 0.0f };
				float lightPower = 10.0f;
				lightSource.power = { lightPower, lightPower, lightPower };
				lightSource.direction = { 0.0f, -1.0f, 0.0f };

				/*lightSource.type = LightSource::SPOT;
				lightSource.position = { 0.0f,-0.25f,0.0f };
				lightSource.power = { 1.0f, 1.0f, 1.0f };*/
				
				lightSourceBuffer.copy(lightSource);

				uint32_t triangle_input_flags[1] =  // One per SBT record for this build input
				{
					//OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
					OPTIX_GEOMETRY_FLAG_NONE
				};

				triangleBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
				triangleBuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
				//triangleBuildInput.triangleArray.vertexStrideInBytes = sizeof(Math::vec3<float>);
				triangleBuildInput.triangleArray.numVertices = box.verticesRepeated.length;
				triangleBuildInput.triangleArray.vertexBuffers = (CUdeviceptr*)& vertices.device;
				triangleBuildInput.triangleArray.flags = triangle_input_flags;
				triangleBuildInput.triangleArray.numSbtRecords = 1;
				triangleBuildInput.triangleArray.sbtIndexOffsetBuffer = 0;
				triangleBuildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
				triangleBuildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
				triangleBuildInput.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;//new in OptiX7.1.0

				accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
				accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

				Buffer temp(Buffer::Device);
				Buffer compation(Buffer::Device);
				OptixAccelBufferSizes GASBufferSizes;
				optixAccelComputeMemoryUsage(context, &accelOptions, &triangleBuildInput, 1, &GASBufferSizes);
				temp.resize(GASBufferSizes.tempSizeInBytes);
				size_t compactedSizeOffset = ((GASBufferSizes.outputSizeInBytes + 7) / 8) * 8;
				compation.resize(compactedSizeOffset + 8);

				OptixAccelEmitDesc emitProperty = {};
				emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
				emitProperty.result = (CUdeviceptr)((char*)compation.device + compactedSizeOffset);

				optixAccelBuild(context, 0,
					&accelOptions, &triangleBuildInput, 1,// num build inputs, which is the num of vertexBuffers pointers
					temp, GASBufferSizes.tempSizeInBytes,
					compation, GASBufferSizes.outputSizeInBytes,
					&GASHandle, &emitProperty, 1);

				size_t compacted_gas_size;
				cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost);
				::printf("Compatcion: %u to %u\n", GASBufferSizes.outputSizeInBytes, compacted_gas_size);
				if (compacted_gas_size < GASBufferSizes.outputSizeInBytes)
				{
					GASOutput.resize(compacted_gas_size);
					// use handle as input and output
					optixAccelCompact(context, 0, GASHandle, GASOutput, compacted_gas_size, &GASHandle);
				}
				else GASOutput.copy(compation);
				paras.handle = GASHandle;
				paras.trans = (TransInfo*)transInfoDevice;
				/*OptixStackSizes stackSizes = { 0 };
				optixUtilAccumulateStackSizes(programGroups[0], &stackSizes);

				uint32_t max_trace_depth = 1;
				uint32_t max_cc_depth = 0;
				uint32_t max_dc_depth = 0;
				uint32_t direct_callable_stack_size_from_traversal;
				uint32_t direct_callable_stack_size_from_state;
				uint32_t continuation_stack_size;
				optixUtilComputeStackSizes(&stackSizes,
					max_trace_depth, max_cc_depth, max_dc_depth,
					&direct_callable_stack_size_from_traversal,
					&direct_callable_stack_size_from_state,
					&continuation_stack_size
				);
				optixPipelineSetStackSize(pipeline,
					direct_callable_stack_size_from_traversal,
					direct_callable_stack_size_from_state,
					continuation_stack_size, 3);*/

				// ray trace stage
				optixSbtRecordPackHeader(rt_raygen, &rt_raygenData);
				rt_raygenDataBuffer.copy(rt_raygenData);

				optixSbtRecordPackHeader(rt_hitRayRadiance, &rt_hitDatas[RayRadiance]);
				rt_hitDatas[RayRadiance].data.normals = (float3*)normals.device;
				rt_hitDatas[RayRadiance].data.kds = (float3*)kds.device;
				rt_hitDatas[RayRadiance].data.lightSource = (LightSource*)lightSourceBuffer.device;
				optixSbtRecordPackHeader(rt_hitShadowRay, &rt_hitDatas[ShadowRay]);
				rt_hitDatas[ShadowRay].data.normals = (float3*)normals.device;
				rt_hitDatas[ShadowRay].data.kds = (float3*)kds.device;
				rt_hitDatas[ShadowRay].data.lightSource = (LightSource*)lightSourceBuffer.device;
				rt_hitDataBuffer.copy(rt_hitDatas, sizeof(rt_hitDatas));

				SbtRecord<int> rt_missDatas[RayCount];
				optixSbtRecordPackHeader(rt_missRayRadiance, &rt_missDatas[RayRadiance]);
				optixSbtRecordPackHeader(rt_missShadowRay, &rt_missDatas[ShadowRay]);
				rt_missDataBuffer.copy(rt_missDatas, sizeof(rt_missDatas));

				rt_sbt.raygenRecord = rt_raygenDataBuffer;
				rt_sbt.missRecordBase = rt_missDataBuffer;
				rt_sbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				rt_sbt.missRecordCount = RayCount;
				rt_sbt.hitgroupRecordBase = rt_hitDataBuffer;
				rt_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<Rt_HitData>);
				rt_sbt.hitgroupRecordCount = RayCount;

				// photon trace stage
				// NOTE: photonMapBuffer should be larger in case of overflow due to the kdTree access
				photonBuffer.resize(sizeof(Photon)* PT_MAX_DEPOSIT* PT_PHOTON_CNT);
				photonMapBuffer.resize(sizeof(Photon)* PT_MAX_DEPOSIT * 2 * PT_PHOTON_CNT);

				srand(time(nullptr));
				cudaMalloc(&paras.randState, PT_PHOTON_CNT * sizeof(curandState));
				initRandom(paras.randState, rand(), 1024, (PT_PHOTON_CNT + 1023) / 1024, PT_PHOTON_CNT);

				optixSbtRecordPackHeader(pt_raygen, &pt_raygenData);
				pt_raygenData.data.lightSource = (LightSource*)lightSourceBuffer.device;
				pt_raygenData.data.photons = (Photon*)photonBuffer.device;
				pt_raygenDataBuffer.copy(pt_raygenData);

				optixSbtRecordPackHeader(pt_closestHit, &pt_hitData);
				pt_hitData.data.normals = (float3*)normals.device;
				pt_hitData.data.kds = (float3*)kds.device;
				pt_hitData.data.photons = (Photon*)photonBuffer.device;
				pt_hitDataBuffer.copy(pt_hitData);

				SbtRecord<int> pt_missData;
				optixSbtRecordPackHeader(pt_miss, &pt_missData);
				pt_missDataBuffer.copy(pt_missData);

				pt_sbt.raygenRecord = pt_raygenDataBuffer;
				pt_sbt.missRecordBase = pt_missDataBuffer;
				pt_sbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				pt_sbt.missRecordCount = 1;
				pt_sbt.hitgroupRecordBase = pt_hitDataBuffer;
				pt_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<Pt_HitData>);
				pt_sbt.hitgroupRecordCount = 1;

				//// debug data
				//// NOTE: resize unfinished!
				//int debugDatasCnt = _size.w * _size.h;
				//DebugData* debugDatasTemp = new DebugData[debugDatasCnt];
				//debugDatas.copy(debugDatasTemp, sizeof(DebugData)* debugDatasCnt);
				//delete[] debugDatasTemp;

				cudaStreamCreate(&cuStream);
			}
			virtual void run()
			{
				frameBuffer.map();
				if (photonFlag == true)
				{
					optixLaunch(pt_pip, cuStream, parasBuffer, sizeof(Parameters), &pt_sbt, PT_PHOTON_CNT, 1, 1);
					createPhotonMap();
					photonFlag = false;
				}
				optixLaunch(rt_pip, cuStream, parasBuffer, sizeof(Parameters), &rt_sbt, paras.size.x, paras.size.y, 1);
				//Gt_debug();
				frameBuffer.unmap();
			}
			void Gt_debug()
			{
				debugDatas.map();
				
				size_t debugDatasSize = debugDatas.size;
				int debugDatasCnt = debugDatasSize / sizeof(DebugData);
				DebugData* debugDatasTemp = new DebugData[debugDatasCnt];
				cudaMemcpy(debugDatasTemp, debugDatas.device, debugDatasSize, cudaMemcpyDeviceToHost);

				delete[] debugDatasTemp;

				debugDatas.unmap();
			}
			virtual void resize(OpenGL::FrameScale const& _size, GLuint _gl)
			{
				frameBuffer.resize(_gl);
				frameBuffer.map();
				paras.image = (float4*)frameBuffer.device;
				paras.size = make_uint2(_size.w, _size.h);
				parasBuffer.copy(paras);
				frameBuffer.unmap();
			}
			void terminate()
			{
				cudaFree(paras.randState);
			}
			void buildKdTree(Photon** photons, int start, int end, int depth, Photon* kdTree,
				int root, float3 bbmin, float3 bbmax)
			{
				if (end == start)	// 0 photon -> Null node
				{
					kdTree[root].axis = PPM_NULL;
					kdTree[root].energy = make_float3(0.0f, 0.0f, 0.0f);
					return;
				}

				if (end - start == 1)	// 1 photon -> leaf
				{
					photons[start]->axis = PPM_LEAF;
					kdTree[root] = *(photons[start]);
					return;
				}

				// choose the split axis
				int axis;
				float3 diag = make_float3(bbmax.x - bbmin.x, bbmax.y - bbmin.y, bbmax.z - bbmin.z);

				if (diag.x > diag.y)
				{
					if (diag.x > diag.z)
						axis = 0;
					else
						axis = 2;
				}
				else
				{
					if (diag.y > diag.z)
						axis = 1;
					else
						axis = 2;
				}

				// choose the root photon
				int median = (start + end) / 2;
				Photon** startAddr = &photons[start];

				switch (axis)
				{
					case 0:
						select(startAddr, 0, 0, end - start - 1, median - start);
						photons[median]->axis = PPM_X;
						break;
					case 1:
						select(startAddr, 1, 0, end - start - 1, median - start);
						photons[median]->axis = PPM_Y;
						break;
					case 2:
						select(startAddr, 2, 0, end - start - 1, median - start);
						photons[median]->axis = PPM_Z;
						break;
				}

				// calculate the bounding box
				float3 rightMin = bbmin;
				float3 leftMax = bbmax;
				float3 midPointPosition = (*photons[median]).position;
				switch (axis)
				{
					case 0:
						rightMin.x = midPointPosition.x;
						leftMax.x = midPointPosition.x;
						break;
					case 1:
						rightMin.y = midPointPosition.y;
						leftMax.y = midPointPosition.y;
						break;
					case 2:
						rightMin.z = midPointPosition.z;
						leftMax.z = midPointPosition.z;
						break;
				}

				// recursively build the KdTree
				kdTree[root] = *(photons[median]);
				buildKdTree(photons, start, median, depth + 1, kdTree, 2 * root + 1, bbmin, leftMax);
				buildKdTree(photons, median + 1, end, depth + 1, kdTree, 2 * root + 2, rightMin, bbmax);
			}
			void createPhotonMap()
			{
				photonBuffer.map();
				photonMapBuffer.map();

				// initialize the photon map
				size_t photonMapBufferSize = photonMapBuffer.size;
				int photonMapBufferCnt = photonMapBufferSize / sizeof(Photon);
				Photon* photonMapData = new Photon[photonMapBufferCnt];
				for (int c0(0); c0 < photonMapBufferCnt; c0++)
				{
					photonMapData[c0].energy = { 0.0f,0.0f,0.0f };
					photonMapData[c0].axis = 0;
				}

				// copy photon data to host
				size_t photonBufferSize = photonBuffer.size;
				int photonBufferCnt = photonBufferSize / sizeof(Photon);
				Photon* photonData = new Photon[photonBufferCnt];
				cudaMemcpy(photonData, photonBuffer.device, photonBufferSize, cudaMemcpyDeviceToHost);

				// get valid data
				int validPhotonCnt = 0;
				Photon** tempPhotons = new Photon* [photonBufferCnt];
				for (int c0(0); c0 < photonBufferCnt; c0++)
					if (photonData[c0].energy.x > 0.0f ||
						photonData[c0].energy.y > 0.0f ||
						photonData[c0].energy.z > 0.0f)
					{
						tempPhotons[validPhotonCnt++] = &photonData[c0];
						/*
						printf("photon #%d: position  (%f,%f,%f)\n", c0, photonData[c0].position.x, photonData[c0].position.y, photonData[c0].position.z);
						printf("           direction (%f,%f,%f)\n", photonData[c0].dir.x, photonData[c0].dir.y, photonData[c0].dir.z);
						printf("           normal    (%f,%f,%f)\n", photonData[c0].normal.x, photonData[c0].normal.y, photonData[c0].normal.z);
						printf("           energy    (%f,%f,%f)\n", photonData[c0].energy.x, photonData[c0].energy.y, photonData[c0].energy.z);
						*/
					}


				printf("photonBufferCnt: %d, valid cnt: %d\n", photonBufferCnt, validPhotonCnt);

				if (validPhotonCnt > photonMapBufferCnt)
					validPhotonCnt = photonMapBufferCnt;

				// compute the bounding box
				float floatMax = std::numeric_limits<float>::max();
				float3 bbmin = make_float3(floatMax, floatMax, floatMax);
				float3 bbmax = make_float3(-floatMax, -floatMax, -floatMax);
				for (int c0(0); c0 < validPhotonCnt; c0++)
				{
					float3 position = (*tempPhotons[c0]).position;
					bbmin.x = fminf(bbmin.x, position.x);
					bbmin.y = fminf(bbmin.y, position.y);
					bbmin.z = fminf(bbmin.z, position.z);
					bbmax.x = fmaxf(bbmax.x, position.x);
					bbmax.y = fmaxf(bbmax.y, position.y);
					bbmax.z = fmaxf(bbmax.z, position.z);
				}

				//printf("bbmin:(%f,%f,%f)\n", bbmin.x, bbmin.y, bbmin.z);
				//printf("bbmax:(%f,%f,%f)\n", bbmax.x, bbmax.y, bbmax.z);

				// build the kdTree
				buildKdTree(tempPhotons, 0, validPhotonCnt, 0, photonMapData, 0, bbmin, bbmax);

				/*for (int i = 0; i < photonMapBufferCnt; i++)
				{
					printf("photonMap[%d].position(%f,%f,%f)\n", i, photonMapData[i].position.x, photonMapData[i].position.y, photonMapData[i].position.z);
					printf("             axis %d\n", photonMapData[i].axis);
				}*/

				// copy result to device
				photonMapBuffer.copy(photonMapData, photonMapBufferSize);

				// update the sbt
				rt_hitDataBuffer.map();
				rt_hitDatas[RayRadiance].data.photonMap = (Photon*)photonMapBuffer.device;
				rt_hitDatas[ShadowRay].data.photonMap = (Photon*)photonMapBuffer.device;
				rt_hitDataBuffer.copy(rt_hitDatas, sizeof(rt_hitDatas));
				rt_hitDataBuffer.unmap();

				// free memory
				delete[] photonData;
				delete[] photonMapData;
				delete[] tempPhotons;

				photonBuffer.unmap();
				photonMapBuffer.unmap();
			}
		};
	}
}
namespace OpenGL
{
	struct PathTracing :OpenGL
	{
		SourceManager sm;
		OptiXDefautRenderer renderer;
		//CUDA::Buffer test;
		CUDA::OptiX::Trans trans;
		CUDA::OptiX::PathTracing pathTracer;
		FrameScale size;
		bool frameSizeChanged;
		PathTracing(FrameScale const& _size)
			:
			sm(),
			renderer(&sm, _size),
			//test(CUDA::Buffer::Device, 4),
			trans({ {60},{0.01,0.9,0.005},{0.006},{0,0,5.0},1400.0 }),
			pathTracer(&sm, &renderer, _size, trans.buffer.device),
			size(_size),
			frameSizeChanged(false)
		{
			/*test.resizeHost();
			*(float*)test.host = 1234.5;
			test.moveToDevice();
			test.freeHost();
			test.moveToHost();
			::printf("---%f\n", *(float*)test.host);*/
			trans.init(_size);
		}
		~PathTracing()
		{
			pathTracer.terminate();
		}
		virtual void init(FrameScale const& _size) override
		{
			renderer.resize(_size);
			trans.resize(_size);
			pathTracer.resize(_size, renderer);
		}
		virtual void run() override
		{
			changeFrameSize();
			trans.operate();
			trans.updated = false;
			pathTracer.run();
			renderer.updated = true;
			renderer.use();
			renderer.run();
		}
		void terminate()
		{
		}
		void changeFrameSize()
		{
			if (frameSizeChanged)
			{
				renderer.resize(size);
				trans.resize(size);
				glFinish();
				pathTracer.resize(size, renderer);
				frameSizeChanged = false;
			}
		}
		virtual void frameSize(int _w, int _h)override
		{
			if (size.w != _w || size.h != _h)
			{
				frameSizeChanged = true;
				size.w = _w;
				size.h = _h;
			}
		}
		virtual void framePos(int, int) override {}
		virtual void frameFocus(int) override {}
		virtual void mouseButton(int _button, int _action, int _mods)override
		{
			switch (_button)
			{
				case GLFW_MOUSE_BUTTON_LEFT:trans.mouse.refreshButton(0, _action); break;
				case GLFW_MOUSE_BUTTON_MIDDLE:trans.mouse.refreshButton(1, _action); break;
				case GLFW_MOUSE_BUTTON_RIGHT:trans.mouse.refreshButton(2, _action); break;
			}
		}
		virtual void mousePos(double _x, double _y)override
		{
			trans.mouse.refreshPos(_x, _y);
		}
		virtual void mouseScroll(double _x, double _y)override
		{
			if (_y != 0.0)
				trans.scroll.refresh(_y);
		}
		virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods) override
		{
			{
				switch (_key)
				{
					case GLFW_KEY_ESCAPE:if (_action == GLFW_PRESS)
						glfwSetWindowShouldClose(_window, true); break;
					case GLFW_KEY_A:trans.key.refresh(0, _action); break;
					case GLFW_KEY_D:trans.key.refresh(1, _action); break;
					case GLFW_KEY_W:trans.key.refresh(2, _action); break;
					case GLFW_KEY_S:trans.key.refresh(3, _action); break;
						/*	case GLFW_KEY_UP:monteCarlo.trans.persp.increaseV(0.02); break;
							case GLFW_KEY_DOWN:monteCarlo.trans.persp.increaseV(-0.02); break;
							case GLFW_KEY_RIGHT:monteCarlo.trans.persp.increaseD(0.01); break;
							case GLFW_KEY_LEFT:monteCarlo.trans.persp.increaseD(-0.01); break;*/
				}
			}
		}
	};
}

int main()
{
	OpenGL::OpenGLInit(4, 5);
	Window::Window::Data winPara
	{
		"PathTracer",
		{
			{1920,1080},
			true,false
		}
	};
	Window::WindowManager wm(winPara);
	OpenGL::PathTracing pathTracer(winPara.size.size);
	wm.init(0, &pathTracer);
	glfwSwapInterval(0);
	FPS fps;
	fps.refresh();
	while (!wm.close())
	{
		wm.pullEvents();
		wm.render();
		wm.swapBuffers();
		fps.refresh();
		fps.printFPSAndFrameTime(2, 3);
		//wm.windows[0].data.setTitle(fps.str);
	}
	return 1;
}
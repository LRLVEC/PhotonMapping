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
			OptixModuleCompileOptions gt_moduleCompileOptions;
			OptixPipelineCompileOptions rt_pipelineCompileOptions;
			OptixPipelineCompileOptions pt_pipelineCompileOptions;
			OptixPipelineCompileOptions gt_pipelineCompileOptions;
			ModuleManager mm;
			OptixProgramGroupOptions rt_programGroupOptions;
			OptixProgramGroupOptions pt_programGroupOptions;
			OptixProgramGroupOptions gt_programGroupOptions;
			Program miss;
			Program rt_raygen;
			Program rt_closestHit;
			Program pt_raygen;
			Program pt_closestHit;
			Program gt_raygen;
			Program gt_closestHit;
			OptixPipelineLinkOptions rt_pipelineLinkOptions;
			OptixPipelineLinkOptions pt_pipelineLinkOptions;
			OptixPipelineLinkOptions gt_pipelineLinkOptions;
			Pipeline rt_pip;
			Pipeline pt_pip;
			Pipeline gt_pip;
			SbtRecord<int> missData;
			SbtRecord<Rt_RayGenData> rt_raygenData;
			SbtRecord<Rt_CloseHitData> rt_hitData;
			SbtRecord<Pt_RayGenData> pt_raygenData;
			SbtRecord<Pt_CloseHitData> pt_hitData;
			SbtRecord<Gt_RayGenData> gt_raygenData;
			SbtRecord<int> gt_hitData;
			LightSource lightSource;
			Buffer lightSourceBuffer;
			Buffer cameraRayHitBuffer;
			Buffer photonBuffer;
			Buffer photonMapBuffer;
			Buffer missDataBuffer;
			Buffer rt_raygenDataBuffer;
			Buffer rt_hitDataBuffer;
			Buffer pt_raygenDataBuffer;
			Buffer pt_hitDataBuffer;
			Buffer gt_raygenDataBuffer;
			Buffer gt_hitDataBuffer;
			OptixShaderBindingTable rt_sbt;
			OptixShaderBindingTable pt_sbt;
			OptixShaderBindingTable gt_sbt;
			Buffer frameBuffer;
			CUstream cuStream;
			Parameters paras;
			Buffer parasBuffer;
			STL box;
			Buffer vertices;
			Buffer normals;
			Buffer kds;
			Buffer pt_emitPositionSeeds;
			Buffer pt_emitDirectionSeeds;
			Buffer pt_hitDirectionSeeds;
			Buffer RRSeeds;
			OptixBuildInput triangleBuildInput;
			OptixAccelBuildOptions accelOptions;
			Buffer GASOutput;
			OptixTraversableHandle GASHandle;
			unsigned int errorCounter;
			bool photonFlag;
			PathTracing(OpenGL::SourceManager* _sourceManager, OpenGL::OptiXDefautRenderer* dr, OpenGL::FrameScale const& _size, void* transInfoDevice)
				:
				context(),
				rt_moduleCompileOptions{
					OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
					OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
					OPTIX_COMPILE_DEBUG_LEVEL_FULL },
				pt_moduleCompileOptions{
					OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
					OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
					OPTIX_COMPILE_DEBUG_LEVEL_FULL },
				gt_moduleCompileOptions{
					OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
					OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
					OPTIX_COMPILE_DEBUG_LEVEL_FULL },
				rt_pipelineCompileOptions{ false,
					OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
					4,2,OPTIX_EXCEPTION_FLAG_NONE,"paras", unsigned int(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE) },//OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE: new in OptiX7.1.0
				pt_pipelineCompileOptions{ false,
					OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
					4,2,OPTIX_EXCEPTION_FLAG_NONE,"paras", unsigned int(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE) },
				gt_pipelineCompileOptions{ false,
					OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
					4,2,OPTIX_EXCEPTION_FLAG_NONE,"paras", unsigned int(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE) },
				mm(&_sourceManager->folder, context, &rt_moduleCompileOptions, &rt_pipelineCompileOptions),
				rt_programGroupOptions{},
				pt_programGroupOptions{},
				gt_programGroupOptions{},
				miss(Vector<String<char>>("__miss__Ahh"), Program::Miss, &rt_programGroupOptions, context, &mm),
				rt_raygen(Vector<String<char>>("__raygen__RayAllocator"), Program::RayGen, &rt_programGroupOptions, context, &mm),
				rt_closestHit(Vector<String<char>>("__closesthit__RayHit"), Program::HitGroup, &rt_programGroupOptions, context, &mm),
				pt_raygen(Vector<String<char>>("__raygen__PhotonEmit"), Program::RayGen, &pt_programGroupOptions, context, &mm),
				pt_closestHit(Vector<String<char>>("__closesthit__PhotonHit"), Program::HitGroup, &pt_programGroupOptions, context, &mm),
				gt_raygen(Vector<String<char>>("__raygen__Gather"), Program::RayGen, &gt_programGroupOptions, context, &mm),
				gt_closestHit(Vector<String<char>>("__closesthit__ShadowRayHit"), Program::HitGroup, &gt_programGroupOptions, context, &mm),
				rt_pipelineLinkOptions{ 1,OPTIX_COMPILE_DEBUG_LEVEL_FULL },//no overrideUsesMotionBlur in OptiX7.1.0
				pt_pipelineLinkOptions{ 10,OPTIX_COMPILE_DEBUG_LEVEL_FULL }, // NOTE: maxDepth = 10 in photon trace stage
				gt_pipelineLinkOptions{ 1,OPTIX_COMPILE_DEBUG_LEVEL_FULL },
				rt_pip(context, &rt_pipelineCompileOptions, &rt_pipelineLinkOptions, { rt_raygen ,rt_closestHit, miss }),
				pt_pip(context, &pt_pipelineCompileOptions, &pt_pipelineLinkOptions, { pt_raygen ,pt_closestHit, miss }),
				gt_pip(context, &gt_pipelineCompileOptions, &gt_pipelineLinkOptions, { gt_raygen ,gt_closestHit, miss }),//NOTE:it still workds replaced by rt_closestHit, why?
				lightSourceBuffer(lightSource, false),
				cameraRayHitBuffer(Buffer::Device),
				photonBuffer(Buffer::Device),
				photonMapBuffer(Buffer::Device),
				missDataBuffer(missData, false),
				rt_raygenDataBuffer(rt_raygenData, false),
				rt_hitDataBuffer(rt_hitData, false),
				pt_raygenDataBuffer(pt_raygenData, false),
				pt_hitDataBuffer(pt_hitData, false),
				gt_raygenDataBuffer(gt_raygenData, false),
				gt_hitDataBuffer(gt_hitData, false),
				rt_sbt({}),
				pt_sbt({}),
				gt_sbt({}),
				frameBuffer(*dr),
				parasBuffer(paras, false),
				box(_sourceManager->folder.find("resources/boxnew.stl").readSTL()),
				vertices(Buffer::Device),
				normals(Buffer::Device),
				kds(Buffer::Device),
				pt_emitPositionSeeds(Buffer::Device),
				pt_emitDirectionSeeds(Buffer::Device),
				pt_hitDirectionSeeds(Buffer::Device),
				RRSeeds(Buffer::Device),
				triangleBuildInput({}),
				accelOptions({}),
				GASOutput(Buffer::Device),
				GASHandle(0),
				errorCounter(0),
				photonFlag(true)
			{
				box.getVerticesRepeated();
				box.getNormals();
				box.printInfo(false);
				vertices.copy(box.verticesRepeated.data, sizeof(Math::vec3<float>)* box.verticesRepeated.length);
				normals.copy(box.normals.data, sizeof(Math::vec3<float>)* box.normals.length);

				float3* kdsTemp = new float3[box.normals.length];
				for (int c0(0); c0 < box.normals.length; c0++)
					kdsTemp[c0] = { 1.0f, 1.0f, 1.0f };
				kdsTemp[20] = make_float3(1.0f, 0.0f, 0.0f);
				kdsTemp[21] = make_float3(1.0f, 0.0f, 0.0f);
				kdsTemp[24] = make_float3(0.0f, 1.0f, 0.0f);
				kdsTemp[25] = make_float3(0.0f, 1.0f, 0.0f);
				kds.copy(kdsTemp, sizeof(float3)* box.normals.length);
				delete[] kdsTemp;

				lightSource.type = LightSource::SQUARE;
				lightSource.position = { -0.3f, 0.99f, -0.3f };
				lightSource.power = { 1.0f, 1.0f, 1.0f };
				lightSource.edge1 = { 0.6f, 0.0f, 0.0f };
				lightSource.edge2 = { 0.0f, 0.0f, 0.6f };
				lightSource.direction = { 0.0f, -1.0f, 0.0f };

				/*lightSource.type = LightSource::SPOT;
				lightSource.position = { 0.0f,-0.25f,0.0f };
				lightSource.power = { 1.0f, 1.0f, 1.0f };*/
				
				lightSourceBuffer.copy(lightSource);

				uint32_t triangle_input_flags[1] =  // One per SBT record for this build input
				{
					OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
					//OPTIX_GEOMETRY_FLAG_NONE
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
				paras.maxPhotonCnt = 8;
				paras.maxDepth = 8;
				paras.pt_size = make_uint2(400, 400);
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
				cameraRayHitBuffer.resize(sizeof(CameraRayHitData)* _size.w* _size.h);

				// ray trace stage
				optixSbtRecordPackHeader(rt_raygen, &rt_raygenData);
				rt_raygenData.data.cameraRayHitDatas = (CameraRayHitData*)cameraRayHitBuffer.device;
				rt_raygenDataBuffer.copy(rt_raygenData);

				optixSbtRecordPackHeader(miss, &missData);
				missDataBuffer.copy(missData);

				optixSbtRecordPackHeader(rt_closestHit, &rt_hitData);
				rt_hitData.data.normals = (float3*)normals.device;
				rt_hitData.data.kds = (float3*)kds.device;
				rt_hitData.data.cameraRayHitDatas = (CameraRayHitData*)cameraRayHitBuffer.device;
				rt_hitDataBuffer.copy(rt_hitData);

				rt_sbt.raygenRecord = rt_raygenDataBuffer;
				rt_sbt.missRecordBase = missDataBuffer;
				rt_sbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				rt_sbt.missRecordCount = 1;
				rt_sbt.hitgroupRecordBase = rt_hitDataBuffer;
				rt_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<Rt_CloseHitData>);
				rt_sbt.hitgroupRecordCount = 1;

				// photon trace stage
				// NOTE: photonMapBuffer should be larger in case of overflow due to the kdTree access
				photonBuffer.resize(sizeof(Photon)* paras.maxPhotonCnt* paras.pt_size.x* paras.pt_size.y);
				photonMapBuffer.resize(sizeof(Photon)* paras.maxPhotonCnt * 2 * paras.pt_size.x * paras.pt_size.y);

				srand((unsigned int)time(0));
				int seedCnt = paras.pt_size.x * paras.pt_size.y;
				float2* seedsTemp = new float2[seedCnt];
				for (int c0(0); c0 < seedCnt; c0++)
					seedsTemp[c0] = make_float2(float(rand() % 10000) / 10000, float(rand() % 10000) / 10000);
				pt_emitPositionSeeds.copy(seedsTemp, sizeof(float2)* seedCnt);

				for (int c0(0); c0 < seedCnt; c0++)
					seedsTemp[c0] = make_float2(float(rand() % 10000) / 10000, float(rand() % 10000) / 10000);
				pt_emitDirectionSeeds.copy(seedsTemp, sizeof(float2)* seedCnt);

				for (int c0(0); c0 < seedCnt; c0++)
					seedsTemp[c0] = make_float2(float(rand() % 10000) / 10000, float(rand() % 10000) / 10000);
				pt_hitDirectionSeeds.copy(seedsTemp, sizeof(float2)* seedCnt);
				delete[] seedsTemp;

				seedCnt = paras.maxDepth * paras.pt_size.x * paras.pt_size.y;
				float* RRseedsTemp = new float[seedCnt];
				for (int c0(0); c0 < seedCnt; c0++)
					RRseedsTemp[c0] = float(rand() % 10000) / 10000;
				RRSeeds.copy(sizeof(float)* seedCnt);
				delete[] RRseedsTemp;

				optixSbtRecordPackHeader(pt_raygen, &pt_raygenData);
				pt_raygenData.data.lightSource = (LightSource*)lightSourceBuffer.device;
				pt_raygenData.data.photons = (Photon*)photonBuffer.device;
				pt_raygenData.data.positionSeeds = (float2*)pt_emitPositionSeeds.device;
				pt_raygenData.data.directionSeeds = (float2*)pt_emitDirectionSeeds.device;
				pt_raygenDataBuffer.copy(pt_raygenData);

				optixSbtRecordPackHeader(pt_closestHit, &pt_hitData);
				pt_hitData.data.normals = (float3*)normals.device;
				pt_hitData.data.kds = (float3*)kds.device;
				pt_hitData.data.photons = (Photon*)photonBuffer.device;
				pt_hitData.data.directionSeeds = (float2*)pt_hitDirectionSeeds.device;
				pt_hitData.data.RRseeds = (float*)RRSeeds.device;
				pt_hitDataBuffer.copy(pt_hitData);

				pt_sbt.raygenRecord = pt_raygenDataBuffer;
				pt_sbt.missRecordBase = missDataBuffer;
				pt_sbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				pt_sbt.missRecordCount = 1;
				pt_sbt.hitgroupRecordBase = pt_hitDataBuffer;
				pt_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<Pt_CloseHitData>);
				pt_sbt.hitgroupRecordCount = 1;

				// gather stage sbt binding
				optixSbtRecordPackHeader(gt_raygen, &gt_raygenData);
				gt_raygenData.data.cameraRayHitDatas = (CameraRayHitData*)cameraRayHitBuffer.device;
				gt_raygenData.data.normals = (float3*)normals.device;
				gt_raygenData.data.kds = (float3*)kds.device;
				gt_raygenData.data.lightSource = (LightSource*)lightSourceBuffer.device;
				gt_raygenDataBuffer.copy(gt_raygenData);

				optixSbtRecordPackHeader(gt_closestHit, &gt_hitData);
				gt_hitDataBuffer.copy(gt_hitData);

				gt_sbt.raygenRecord = gt_raygenDataBuffer;
				gt_sbt.missRecordBase = missDataBuffer;
				gt_sbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				gt_sbt.missRecordCount = 1;
				gt_sbt.hitgroupRecordBase = gt_hitDataBuffer;
				gt_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<int>);
				gt_sbt.hitgroupRecordCount = 1;

				cudaStreamCreate(&cuStream);
			}
			virtual void run()
			{
				frameBuffer.map();
				optixLaunch(rt_pip, cuStream, parasBuffer, sizeof(Parameters), &rt_sbt, paras.size.x, paras.size.y, 1);
				if (photonFlag == true)
				{
					optixLaunch(pt_pip, cuStream, parasBuffer, sizeof(Parameters), &pt_sbt, paras.pt_size.x, paras.pt_size.y, 1);
					createPhotonMap();
					photonFlag = false;
				}
				optixLaunch(gt_pip, cuStream, parasBuffer, sizeof(Parameters), &gt_sbt, paras.size.x, paras.size.y, 1);
				frameBuffer.unmap();
			}
			virtual void resize(OpenGL::FrameScale const& _size, GLuint _gl)
			{
				::printf("%u\n", ++errorCounter);
				//maybe I can avoid adjusting this frequently...
				//which means after changing the frame size, don't adjust this at once
				//but wait for one more frame to check if the changing is finished yet...
				frameBuffer.resize(_gl);
				frameBuffer.map();
				paras.image = (float4*)frameBuffer.device;
				paras.size = make_uint2(_size.w, _size.h);
				parasBuffer.copy(paras);
				frameBuffer.unmap();

				// resize the cameraRayHitBuffer
				cameraRayHitBuffer.resize(sizeof(CameraRayHitData) * _size.w * _size.h);

				rt_raygenDataBuffer.map();
				rt_raygenData.data.cameraRayHitDatas = (CameraRayHitData*)cameraRayHitBuffer.device;
				rt_raygenDataBuffer.copy(rt_raygenData);
				rt_raygenDataBuffer.unmap();

				rt_hitDataBuffer.map();
				rt_hitData.data.cameraRayHitDatas = (CameraRayHitData*)cameraRayHitBuffer.device;
				rt_hitDataBuffer.copy(rt_hitData);
				rt_hitDataBuffer.unmap();

				gt_raygenDataBuffer.map();
				gt_raygenData.data.cameraRayHitDatas = (CameraRayHitData*)cameraRayHitBuffer.device;
				gt_raygenDataBuffer.copy(gt_raygenData);
				gt_raygenDataBuffer.unmap();
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
						//printf("photon #%d: position  (%f,%f,%f)\n", c0, photonData[c0].position.x, photonData[c0].position.y, photonData[c0].position.z);
						//printf("           direction (%f,%f,%f)\n", photonData[c0].dir.x, photonData[c0].dir.y, photonData[c0].dir.z);
						//printf("           normal    (%f,%f,%f)\n", photonData[c0].normal.x, photonData[c0].normal.y, photonData[c0].normal.z);
						//printf("           energy    (%f,%f,%f)\n", photonData[c0].energy.x, photonData[c0].energy.y, photonData[c0].energy.z);
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
				gt_raygenData.data.photonMap = (Photon*)photonMapBuffer.device;
				gt_raygenDataBuffer.copy(gt_raygenData);
				gt_raygenDataBuffer.unmap();

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
		PathTracing(FrameScale const& _size)
			:
			sm(),
			renderer(&sm, _size),
			//test(CUDA::Buffer::Device, 4),
			trans({ {60},{0.01,0.9,0.005},{0.006},{0,0,0},1400.0 }),
			pathTracer(&sm, &renderer, _size, trans.buffer.device)
		{
			/*test.resizeHost();
			*(float*)test.host = 1234.5;
			test.moveToDevice();
			test.freeHost();
			test.moveToHost();
			::printf("---%f\n", *(float*)test.host);*/
			trans.init(_size);
		}
		virtual void init(FrameScale const& _size) override
		{
			renderer.resize(_size);
			trans.resize(_size);
			pathTracer.resize(_size, renderer);
		}
		virtual void run() override
		{
			trans.operate();
			if (trans.updated)
			{
				trans.updated = false;
			}
			pathTracer.run();
			renderer.updated = true;
			renderer.use();
			renderer.run();
		}
		void terminate()
		{
		}
		virtual void frameSize(int _w, int _h)override
		{
			renderer.resize({ _w,_h });
			trans.resize({ _w,_h });
			glFinish();
			pathTracer.resize({ _w,_h }, renderer);
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
			{800,600},
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
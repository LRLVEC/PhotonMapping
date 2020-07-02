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

template<class Elem> __inline void swap(Elem* list, int a, int b)
{
	Elem temp = list[a];
	list[a] = list[b];
	list[b] = temp;
}

template<class Elem, int axis> int partition(Elem* list, int left, int right, int pivotIndex)
{
	Elem pivotValue = list[pivotIndex];
	swap(list, right, pivotIndex);
	pivotIndex = right;
	left--;
	while (1) {
		do {
			left++;
		} while (left < right && ElemIndex(list[left], axis) < ElemIndex(pivotValue, axis));
		do {
			right--;
		} while (left < right && ElemIndex(list[right], axis) > ElemIndex(pivotValue, axis));
		if (left < right) {
			swap(list, left, right);
		}
		else {
			// Put the pivotValue back in place
			swap(list, left, pivotIndex);
			return left;
		}
	}
}

template<class Elem, int axis> Elem select(Elem* list, int left, int right, int k)
{
	while (true)
	{
		// select a value to pivot around between left and right and store the index to it.
		int pivotIndex = (left + right) / 2;
		// Determine where this value ended up.
		int pivotNewIndex = partition<Elem, axis>(list, left, right, pivotIndex);
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
			OptixModuleCompileOptions gather_moduleCompileOptions;
			OptixPipelineCompileOptions rt_pipelineCompileOptions;
			OptixPipelineCompileOptions pt_pipelineCompileOptions;
			OptixPipelineCompileOptions gather_pipelineCompileOptions;
			ModuleManager mm;
			OptixProgramGroupOptions rt_programGroupOptions;
			OptixProgramGroupOptions pt_programGroupOptions;
			OptixProgramGroupOptions gather_programGroupOptions;
			Program rt_rayAllocator;
			Program miss;
			Program rt_closestHit;
			Program pt_photonEmit;
			Program pt_photonHit;
			Program gather_gather;
			Program gather_shadowRayHit;
			OptixPipelineLinkOptions rt_pipelineLinkOptions;
			OptixPipelineLinkOptions pt_pipelineLinkOptions;
			OptixPipelineLinkOptions gather_pipelineLinkOptions;
			Pipeline rt_pip;	// ray tracing pipeline
			Pipeline pt_pip;	// photon tracing pipeline
			Pipeline gather_pip;	// gather pipeline
			SbtRecord<int> missData;
			SbtRecord<Rt_RayGenData> rt_raygenData;
			SbtRecord<Rt_CloseHitData> rt_hitData;
			SbtRecord<Pt_RayGenData> pt_raygenData;
			SbtRecord<Pt_CloseHitData> pt_hitData;
			SbtRecord<Gather_RayGenData> gather_raygenData;
			SbtRecord<Gather_CloseHitData> gather_hitData;
			LightSource lightSource;
			Buffer lightSourceBuffer;
			Buffer missDataBuffer;
			Buffer rt_raygenDataBuffer;
			Buffer rt_hitDataBuffer;
			Buffer cameraRayHitBuffer;
			Buffer pt_photonBuffer;
			Buffer pt_raygenDataBuffer;
			Buffer pt_hitDataBuffer;
			Buffer gather_raygenDataBuffer;
			Buffer gather_hitDataBuffer;
			Buffer photonMap;
			OptixShaderBindingTable rt_sbt;
			OptixShaderBindingTable pt_sbt;
			OptixShaderBindingTable gather_sbt;
			Buffer frameBuffer;
			CUstream cuStream;
			Parameters paras;
			Buffer parasBuffer;
			STL box;
			Buffer vertices;
			Buffer normals;
			Buffer colors;
			Buffer kd;
			Buffer attenKd;
			OptixBuildInput triangleBuildInput;
			OptixAccelBuildOptions accelOptions;
			Buffer GASOutput;
			OptixTraversableHandle GASHandle;
			PathTracing(OpenGL::SourceManager* _sourceManager, OpenGL::OptiXDefautRenderer* dr, OpenGL::FrameScale const& _size, void* transInfoDevice)
				:
				context(),
				rt_moduleCompileOptions{ OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,OPTIX_COMPILE_DEBUG_LEVEL_FULL },
				pt_moduleCompileOptions{ OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,OPTIX_COMPILE_DEBUG_LEVEL_FULL },
				gather_moduleCompileOptions{ OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,OPTIX_COMPILE_DEBUG_LEVEL_FULL },
				rt_pipelineCompileOptions{ false,OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,3,2,OPTIX_EXCEPTION_FLAG_DEBUG,"paras" },
				pt_pipelineCompileOptions{ false,OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,3,2,OPTIX_EXCEPTION_FLAG_DEBUG,"paras" },
				gather_pipelineCompileOptions{ false,OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,3,2,OPTIX_EXCEPTION_FLAG_DEBUG,"paras" },
				mm(&_sourceManager->folder, context, &rt_moduleCompileOptions, &rt_pipelineCompileOptions),
				rt_programGroupOptions{},
				pt_programGroupOptions{},
				gather_programGroupOptions{},
				rt_rayAllocator(Vector<String<char>>("__raygen__RayAllocator"), Program::RayGen, &rt_programGroupOptions, context, &mm),
				miss(Vector<String<char>>("__miss__Ahh"), Program::Miss, &rt_programGroupOptions, context, &mm),
				rt_closestHit(Vector<String<char>>("__closesthit__RayHit"), Program::HitGroup, &rt_programGroupOptions, context, &mm),
				pt_photonEmit(Vector<String<char>>("__raygen__PhotonEmit"), Program::RayGen, &pt_programGroupOptions, context, &mm),
				pt_photonHit(Vector<String<char>>("__closesthit__PhotonHit"), Program::HitGroup, &pt_programGroupOptions, context, &mm),
				gather_gather(Vector<String<char>>("__raygen__Gather"), Program::RayGen, &gather_programGroupOptions, context, &mm),
				gather_shadowRayHit(Vector<String<char>>("__closesthit__ShadowRayHit"), Program::HitGroup, &gather_programGroupOptions, context, &mm),
				rt_pipelineLinkOptions{ 2,OPTIX_COMPILE_DEBUG_LEVEL_FULL,false },
				pt_pipelineLinkOptions{ 2,OPTIX_COMPILE_DEBUG_LEVEL_FULL,false },
				gather_pipelineLinkOptions{ 2,OPTIX_COMPILE_DEBUG_LEVEL_FULL,false },
				rt_pip(context, &rt_pipelineCompileOptions, &rt_pipelineLinkOptions, { rt_rayAllocator ,rt_closestHit, miss }),
				pt_pip(context, &pt_pipelineCompileOptions, &pt_pipelineLinkOptions, { pt_photonEmit ,pt_photonHit ,miss }),
				gather_pip(context, &gather_pipelineCompileOptions, &gather_pipelineLinkOptions, { gather_gather, gather_shadowRayHit,miss }),
				missDataBuffer(missData, false),
				lightSourceBuffer(lightSource, false),
				rt_raygenDataBuffer(rt_raygenData, false),
				rt_hitDataBuffer(rt_hitData, false),
				cameraRayHitBuffer(Buffer::Device),
				pt_photonBuffer(Buffer::Device),
				pt_raygenDataBuffer(pt_raygenData, false),
				pt_hitDataBuffer(pt_hitData, false),
				gather_raygenDataBuffer(gather_raygenData, false),
				gather_hitDataBuffer(gather_hitData, false),
				photonMap(Buffer::Device),
				rt_sbt(),
				pt_sbt(),
				gather_sbt(),
				frameBuffer(*dr),
				parasBuffer(paras, false),
				box(_sourceManager->folder.find("resources/teapot.stl").readSTL()),
				vertices(Buffer::Device),
				normals(Buffer::Device),
				colors(Buffer::Device),
				kd(Buffer::Device),
				attenKd(Buffer::Device),
				triangleBuildInput({}),
				accelOptions({}),
				GASOutput(Buffer::Device)
			{
				box.getVerticesRepeated();
				box.getNormals();
				box.printInfo(false);
				vertices.copy(box.verticesRepeated.data, sizeof(Math::vec3<float>)* box.verticesRepeated.length);
				normals.copy(box.normals.data, sizeof(Math::vec3<float>)* box.normals.length);

				float3* colorsTemp = (float3*)malloc(sizeof(float3) * box.normals.length);
				for (int c0(0); c0 < box.normals.length; c0++)
					colorsTemp[c0] = make_float3(0.462f, 0.725f, 0.f);
				colors.copy(colorsTemp, sizeof(float3)* box.normals.length);
				free(colorsTemp);

				float3* kdTemp = (float3*)malloc(sizeof(float3) * box.normals.length);
				for (int c0(0); c0 < box.normals.length; c0++)
					kdTemp[c0] = make_float3(0.7f, 0.7f, 0.7f);
				kd.copy(kdTemp, sizeof(float3)* box.normals.length);
				free(kdTemp);

				float3* attenKdTemp = (float3*)malloc(sizeof(float3) * box.normals.length);
				for (int c0(0); c0 < box.normals.length; c0++)
					attenKdTemp[c0] = make_float3(1.0f, 1.0f, 1.0f);
				attenKd.copy(attenKdTemp, sizeof(float3)* box.normals.length);
				free(attenKdTemp);

				uint32_t triangle_input_flags[1] =  // One per SBT record for this build input
				{
					OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
				};

				triangleBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
				triangleBuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
				triangleBuildInput.triangleArray.vertexStrideInBytes = sizeof(Math::vec3<float>);
				triangleBuildInput.triangleArray.numVertices = box.verticesRepeated.length;
				triangleBuildInput.triangleArray.vertexBuffers = (CUdeviceptr*)&vertices.device;
				triangleBuildInput.triangleArray.flags = triangle_input_flags;
				triangleBuildInput.triangleArray.numSbtRecords = 1;
				triangleBuildInput.triangleArray.sbtIndexOffsetBuffer = 0;
				triangleBuildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
				triangleBuildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

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
					// use handle as input and output-
					optixAccelCompact(context, 0, GASHandle, GASOutput, compacted_gas_size, &GASHandle);
				}
				else GASOutput.copy(compation);
				paras.handle = GASHandle;
				paras.trans = (TransInfo*)transInfoDevice;
				paras.maxPhotonCnt = 5;
				paras.maxDepth = 5;
				paras.pt_size = make_uint2(20, 20);
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
					// ray trace sbt binding
				cameraRayHitBuffer.resize(sizeof(CameraRayHitData)* _size.w* _size.h);

				optixSbtRecordPackHeader(rt_rayAllocator, &rt_raygenData);
				rt_raygenData.data.rayData = { 1.f, 1.f, 1.f };
				rt_raygenData.data.cameraRayHitData = (CameraRayHitData*)cameraRayHitBuffer.device;
				rt_raygenDataBuffer.copy(rt_raygenData);

				optixSbtRecordPackHeader(miss, &missData);
				missDataBuffer.copy(missData);

				optixSbtRecordPackHeader(rt_closestHit, &rt_hitData);
				rt_hitData.data.normals = (float3*)normals.device;
				rt_hitData.data.colors = (float3*)colors.device;
				rt_hitData.data.kd = (float3*)kd.device;
				rt_hitData.data.cameraRayHitData = (CameraRayHitData*)cameraRayHitBuffer.device;
				rt_hitDataBuffer.copy(rt_hitData);

				rt_sbt.raygenRecord = rt_raygenDataBuffer;
				rt_sbt.missRecordBase = missDataBuffer;
				rt_sbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				rt_sbt.missRecordCount = 1;
				rt_sbt.hitgroupRecordBase = rt_hitDataBuffer;
				rt_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<Rt_CloseHitData>);
				rt_sbt.hitgroupRecordCount = 1;

				// photon trace sbt binding
				lightSource.position = { 5.0f, 5.0f, 5.0f };
				lightSource.direction = make_float3(-1.0f, -1.0f, -1.0f);
				lightSource.power = { 1.0f,1.0f,1.0f };
				lightSource.type = LightSource::SPOT;
				lightSourceBuffer.copy(lightSource);

				pt_photonBuffer.resize(sizeof(PhotonRecord)* paras.maxPhotonCnt* paras.pt_size.x* paras.pt_size.y);
				photonMap.resize(sizeof(PhotonRecord)* paras.maxPhotonCnt* paras.pt_size.x* paras.pt_size.y);

				optixSbtRecordPackHeader(pt_photonEmit, &pt_raygenDataBuffer);
				pt_raygenData.data.lightSource = (LightSource*)lightSourceBuffer.device;
				pt_raygenData.data.photonRecord = (PhotonRecord*)pt_photonBuffer.device;
				pt_raygenDataBuffer.copy(pt_raygenData);

				optixSbtRecordPackHeader(pt_photonHit, &pt_hitDataBuffer);
				pt_hitData.data.normals = (float3*)normals.device;
				pt_hitData.data.colors = (float3*)colors.device;
				pt_hitData.data.kd = (float3*)kd.device;
				pt_hitData.data.photonRecord = (PhotonRecord*)pt_photonBuffer.device;
				pt_hitDataBuffer.copy(pt_hitData);

				pt_sbt.raygenRecord = pt_raygenDataBuffer;
				pt_sbt.missRecordBase = missDataBuffer;
				pt_sbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				pt_sbt.missRecordCount = 1;
				pt_sbt.hitgroupRecordBase = pt_hitDataBuffer;
				pt_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<PhotonRecord>);
				pt_sbt.hitgroupRecordCount = 1;

				// gather stage sbt binding
				optixSbtRecordPackHeader(gather_gather, &gather_raygenData);
				gather_raygenData.data.cameraRayHitData = (CameraRayHitData*)cameraRayHitBuffer.device;
				gather_raygenData.data.photonMap = (PhotonRecord*)photonMap.device;
				gather_raygenData.data.normals = (float3*)normals.device;
				gather_raygenData.data.colors = (float3*)colors.device;
				gather_raygenData.data.kd = (float3*)kd.device;
				gather_raygenData.data.attenKd = (float3*)attenKd.device;
				gather_raygenData.data.lightSource = (LightSource*)lightSourceBuffer.device;
				gather_raygenDataBuffer.copy(gather_raygenData);

				optixSbtRecordPackHeader(gather_shadowRayHit, &gather_hitData);
				gather_hitData.data.normals = (float3*)normals.device;
				gather_hitDataBuffer.copy(gather_hitData);

				gather_sbt.raygenRecord = gather_raygenDataBuffer;
				gather_sbt.missRecordBase = missDataBuffer;
				gather_sbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				gather_sbt.missRecordCount = 1;
				gather_sbt.hitgroupRecordBase = gather_hitDataBuffer;
				gather_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<Gather_CloseHitData>);
				gather_sbt.hitgroupRecordCount = 1;

				cudaStreamCreate(&cuStream);
			}
			virtual void run()
			{
				frameBuffer.map();
				//optixLaunch(rt_pip, cuStream, parasBuffer, sizeof(Parameters), &rt_sbt, paras.size.x, paras.size.y, 1);
				//optixLaunch(pt_pip, cuStream, parasBuffer, sizeof(Parameters), &pt_sbt, paras.pt_size.x, paras.pt_size.y, 1);
				//createPhotonMap(pt_photonBuffer, photonMap);
				optixLaunch(gather_pip, cuStream, parasBuffer, sizeof(Parameters), &gather_sbt, paras.size.x, paras.size.y, 1);
				frameBuffer.unmap();
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
			void buildKDTree(PhotonRecord** photons, int start, int end, int depth, PhotonRecord* kdTree,
				int currentRoot, float3 bbmin, float3 bbmax)
			{
				// 0 photon -> NULL node
				if (end == start)
				{
					kdTree[currentRoot].axis = PPM_NULL;
					kdTree[currentRoot].energy = make_float3(0.0f, 0.0f, 0.0f);
					return;
				}

				// 1 photon -> Leaf
				if (end - start == 1)
				{
					photons[start]->axis = PPM_LEAF;
					kdTree[currentRoot] = *(photons[start]);
					return;
				}

				// choose the split axis
				int axis;
				float3 diag;
				diag.x = bbmax.x - bbmin.x;
				diag.y = bbmax.y - bbmin.y;
				diag.z = bbmax.z - bbmin.z;
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
				PhotonRecord** startAddr = &photons[start];

				switch (axis)
				{
				case 0:
					select<PhotonRecord*, 0>(startAddr, 0, end - start - 1, median - start);
					photons[median]->axis = PPM_X;
					break;
				case 1:
					select<PhotonRecord*, 1>(startAddr, 0, end - start - 1, median - start);
					photons[median]->axis = PPM_Y;
					break;
				case 2:
					select<PhotonRecord*, 2>(startAddr, 0, end - start - 1, median - start);
					photons[median]->axis = PPM_Z;
					break;
				}

				// calculate the bounding box
				float3 rightMin = bbmin;
				float3 leftMax = bbmax;
				float3 midPoint = (*photons[median]).position;
				switch (axis)
				{
				case 0:
					rightMin.x = midPoint.x;
					leftMax.x = midPoint.x;
					break;
				case 1:
					rightMin.y = midPoint.y;
					leftMax.y = midPoint.y;
					break;
				case 2:
					rightMin.z = midPoint.z;
					leftMax.z = midPoint.z;
					break;
				}

				// recursively build the KDTree
				kdTree[currentRoot] = *(photons[median]);
				buildKDTree(photons, start, median, depth + 1, kdTree, 2 * currentRoot + 1, bbmin, leftMax);
				buildKDTree(photons, start, median, depth + 1, kdTree, 2 * currentRoot + 2, bbmin, leftMax);
			}
			void createPhotonMap(Buffer photonBuffer, Buffer photonMapBuffer)
			{
				// initialize the photon map
				size_t photonMapBufferSize = photonMapBuffer.size;
				int photonMapBufferCnt = photonMapBufferSize / sizeof(PhotonRecord);
				PhotonRecord* photonMapData = new PhotonRecord[photonMapBufferCnt];
				float3 zero = make_float3(0.0f, 0.0f, 0.0f);
				for (int c0(0); c0 < (int)photonMapBufferCnt; c0++)
					photonMapData[c0].energy = zero;
				photonMapBuffer.copy(photonMapData, photonMapBufferSize);

				// get valid photons
				size_t photonBufferSize = photonBuffer.size;
				int photonBufferCnt = photonBufferSize / sizeof(PhotonRecord);
				PhotonRecord* photonData = new PhotonRecord[photonBufferCnt];
				cudaMemcpy(photonData, photonBuffer.map(), photonBufferSize, cudaMemcpyDeviceToHost);
				int validPhotonCnt = 0;
				PhotonRecord** tempPhotons = (PhotonRecord**) new PhotonRecord * [photonBufferCnt];
				for (int c0(0); c0 < (int)photonBufferCnt; c0++)
					if (photonData[c0].energy.x > 0.0f ||
						photonData[c0].energy.y > 0.0f ||
						photonData[c0].energy.z > 0.0f)
						tempPhotons[validPhotonCnt++] = &photonData[c0];

				if (validPhotonCnt > photonMapBufferCnt)
					validPhotonCnt = photonMapBufferCnt;

				// compute the bounds of the photons
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

				// build the KDTree
				buildKDTree(tempPhotons, 0, validPhotonCnt, 0, photonMapData, 0, bbmin, bbmax);
				photonMapBuffer.copy(photonMapData, photonMapBufferSize);

				delete[] tempPhotons;
				delete[] photonMapData;
				delete[] photonData;
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
			trans({ {60},{0.01,0.9,0.005},{0.06},{0,0,0},1400.0 }),
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
			{640,360},
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
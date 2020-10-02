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
void initNOLT(int*);
void Gather(CameraRayData* cameraRayDatas, Photon* photonMap, float3* normals, float3* kds, int* photonMapStartIdxs, uint2 size, Parameters& paras);

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
			Buffer cameraRayBuffer;
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
			Buffer NOLT;
			Buffer photonMapStartIdxs;
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
								cameraRayBuffer(Buffer::Device),
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
								NOLT(Buffer::Device),
								photonMapStartIdxs(Buffer::Device),
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

					// debug data
					// NOTE: resize unfinished!
				int debugDatasCnt = _size.w * _size.h;
				DebugData* debugDatasTemp = new DebugData[debugDatasCnt];
				debugDatas.copy(debugDatasTemp, sizeof(DebugData)* debugDatasCnt);
				delete[] debugDatasTemp;

				cameraRayBuffer.resize(sizeof(CameraRayData)* paras.size.x* paras.size.y);

				// ray trace pass
				optixSbtRecordPackHeader(rt_raygen, &rt_raygenData);
				rt_raygenData.data.cameraRayDatas = (CameraRayData*)cameraRayBuffer.device;
				rt_raygenDataBuffer.copy(rt_raygenData);

				optixSbtRecordPackHeader(rt_hitRayRadiance, &rt_hitDatas[RayRadiance]);
				rt_hitDatas[RayRadiance].data.normals = (float3*)normals.device;
				rt_hitDatas[RayRadiance].data.kds = (float3*)kds.device;
				rt_hitDatas[RayRadiance].data.lightSource = (LightSource*)lightSourceBuffer.device;
				rt_hitDatas[RayRadiance].data.cameraRayDatas = (CameraRayData*)cameraRayBuffer.device;
				//rt_hitDatas[RayRadiance].data.debugDatas = (DebugData*)debugDatas.device;
				optixSbtRecordPackHeader(rt_hitShadowRay, &rt_hitDatas[ShadowRay]);
				rt_hitDatas[ShadowRay].data.normals = (float3*)normals.device;
				rt_hitDatas[ShadowRay].data.kds = (float3*)kds.device;
				rt_hitDatas[ShadowRay].data.lightSource = (LightSource*)lightSourceBuffer.device;
				rt_hitDatas[ShadowRay].data.cameraRayDatas = (CameraRayData*)cameraRayBuffer.device;
				//rt_hitDatas[ShadowRay].data.debugDatas = (DebugData*)debugDatas.device;
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

				// photon trace pass
				photonBuffer.resize(sizeof(Photon)* PT_MAX_DEPOSIT* PT_PHOTON_CNT);

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
				Gather((CameraRayData*)cameraRayBuffer.device, (Photon*)photonMapBuffer.device,
					(float3*)normals.device, (float3*)kds.device, (int*)photonMapStartIdxs.device, paras.size, *(Parameters*)parasBuffer.device);
				//Debug();
				frameBuffer.unmap();
			}
			void Debug()
			{
				debugDatas.map();

				size_t debugDatasSize = debugDatas.size;
				int debugDatasCnt = debugDatasSize / sizeof(DebugData);
				DebugData* debugDatasTemp = new DebugData[debugDatasCnt];
				cudaMemcpy(debugDatasTemp, debugDatas.device, debugDatasSize, cudaMemcpyDeviceToHost);

				for (int i = 0; i < 1; i++)
					printf("(%f,%f,%f) -> %d should be %d\n", debugDatasTemp[i].position.x, debugDatasTemp[i].position.y, debugDatasTemp[i].position.z, debugDatasTemp[i].hashValue, hash(debugDatasTemp[i].position));

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

				rt_raygenDataBuffer.map();
				rt_hitDataBuffer.map();
				cameraRayBuffer.resize(sizeof(CameraRayData) * _size.w * _size.h);
				rt_raygenData.data.cameraRayDatas = (CameraRayData*)cameraRayBuffer.device;
				rt_raygenDataBuffer.copy(rt_raygenData);
				rt_hitDatas[RayRadiance].data.cameraRayDatas = (CameraRayData*)cameraRayBuffer.device;
				rt_hitDatas[ShadowRay].data.cameraRayDatas = (CameraRayData*)cameraRayBuffer.device;
				rt_hitDataBuffer.copy(rt_hitDatas, sizeof(rt_hitDatas));
				rt_raygenDataBuffer.unmap();
				rt_hitDataBuffer.unmap();
			}
			void terminate()
			{
				cudaFree(paras.randState);
			}
			void createPhotonMap()
			{
				photonBuffer.map();
				photonMapBuffer.map();

				// copy photon data to host
				size_t photonBufferSize = photonBuffer.size;
				int photonBufferCnt = photonBufferSize / sizeof(Photon);
				Photon* photonData = new Photon[photonBufferCnt];
				cudaMemcpy(photonData, photonBuffer.device, photonBufferSize, cudaMemcpyDeviceToHost);

				// get valid data and compute bounding box
				float floatMax = std::numeric_limits<float>::max();
				float3 bbmin = make_float3(floatMax, floatMax, floatMax);
				float3 bbmax = make_float3(-floatMax, -floatMax, -floatMax);

				int validPhotonCnt = 0;
				PhotonHash* tempPhotons = new PhotonHash[photonBufferCnt];
				for (int c0(0); c0 < photonBufferCnt; c0++)
					if (photonData[c0].energy.x > 0.0f ||
						photonData[c0].energy.y > 0.0f ||
						photonData[c0].energy.z > 0.0f)
					{
						float3 position = photonData[c0].position;

						tempPhotons[validPhotonCnt++].pointer = &photonData[c0];

						bbmin.x = fminf(bbmin.x, position.x);
						bbmin.y = fminf(bbmin.y, position.y);
						bbmin.z = fminf(bbmin.z, position.z);
						bbmax.x = fmaxf(bbmax.x, position.x);
						bbmax.y = fmaxf(bbmax.y, position.y);
						bbmax.z = fmaxf(bbmax.z, position.z);
					}

				printf("photonBufferCnt: %d, valid cnt: %d\n", photonBufferCnt, validPhotonCnt);

				bbmin.x -= 0.001f;
				bbmin.y -= 0.001f;
				bbmin.z -= 0.001f;
				bbmax.x += 0.001f;
				bbmax.y += 0.001f;
				bbmax.z += 0.001f;

				/*printf("bbmin:(%f,%f,%f)\n", bbmin.x, bbmin.y, bbmin.z);
				printf("bbmax:(%f,%f,%f)\n", bbmax.x, bbmax.y, bbmax.z);*/

				// specify the grid size
				paras.gridSize.x = (int)ceilf((bbmax.x - bbmin.x) / HASH_GRID_SIDELENGTH) + 2;
				paras.gridSize.y = (int)ceilf((bbmax.y - bbmin.y) / HASH_GRID_SIDELENGTH) + 2;
				paras.gridSize.z = (int)ceilf((bbmax.z - bbmin.z) / HASH_GRID_SIDELENGTH) + 2;
				/*printf("gridSize:(%d,%d,%d)\n", paras.gridSize.x, paras.gridSize.y, paras.gridSize.z);*/

				// specify the world origin
				paras.gridOrigin.x = bbmin.x - HASH_GRID_SIDELENGTH;
				paras.gridOrigin.y = bbmin.y - HASH_GRID_SIDELENGTH;
				paras.gridOrigin.z = bbmin.z - HASH_GRID_SIDELENGTH;
				/*printf("gridOrigin:(%f,%f,%f)\n", paras.gridOrigin.x, paras.gridOrigin.y, paras.gridOrigin.z);
*/
				parasBuffer.copy(paras);

				// compute hash value
				for (int c0(0); c0 < validPhotonCnt; c0++)
					tempPhotons[c0].hashValue = hash(tempPhotons[c0].pointer->position);

				// sort according to hash value
				qsort(tempPhotons, 0, validPhotonCnt);

				// create neighbour offset lookup table
				int* NOLTDatas = new int[27];
				float3 offset[27] =
				{ {0,0,0},{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},
					{-1,-1,0},{-1,1,0},{1,-1,0},{1,1,0},{0,-1,-1},{0,-1,1},{0,1,-1},
					{0,1,1},{-1,0,-1},{-1,0,1},{1,0,-1},{1,0,1},{-1,-1,-1},{-1,-1,1},
					{-1,1,-1},{-1,1,1},{1,-1,-1},{1,-1,1},{1,1,-1},{1,1,1} };
				for (int c0(0); c0 < 27; c0++)
					NOLTDatas[c0] = offset[c0].z * paras.gridSize.x * paras.gridSize.y + offset[c0].y * paras.gridSize.x + offset[c0].x;
				initNOLT(NOLTDatas);
				//NOLT.copy(NOLTDatas, sizeof(int) * 27);
				delete[] NOLTDatas;

				// reorder to build the photonMap
				Photon* photonMapData = new Photon[validPhotonCnt];
				for (int c0(0); c0 < validPhotonCnt; c0++)
					photonMapData[c0] = *(tempPhotons[c0].pointer);
				photonMapBuffer.copy(photonMapData, sizeof(Photon) * validPhotonCnt);

				// find the start index for each cell
				int gridCnt = paras.gridSize.x * paras.gridSize.y * paras.gridSize.z;
				int* startIdxs = new int[gridCnt + 1];
				int i = 0;	// for startIdxs
				int j = 0;	// for tempPhotons
				while (i <= gridCnt)
				{
					if (i < tempPhotons[j].hashValue)
					{
						startIdxs[i++] = j;
						continue;
					}
					if (i == tempPhotons[j].hashValue)
					{
						startIdxs[i] = j;
						while (i == tempPhotons[j].hashValue && j < validPhotonCnt)
							j++;
						i++;
					}
					if (j == validPhotonCnt)
					{
						while (i <= gridCnt)
							startIdxs[i++] = j;
					}
				}
				photonMapStartIdxs.copy(startIdxs, sizeof(int) * (gridCnt + 1));

				/*int emptyCnt = 0;
				for (int i = 0; i < gridCnt; i++)
					if (startIdxs[i] == startIdxs[i + 1])
						emptyCnt++;
				printf("empty %f\%\n", (float)emptyCnt / gridCnt);*/

				delete[] startIdxs;

				// update the sbt
				rt_hitDataBuffer.map();
				rt_hitDatas[RayRadiance].data.photonMap = (Photon*)photonMapBuffer.device;
				rt_hitDatas[ShadowRay].data.photonMap = (Photon*)photonMapBuffer.device;
				rt_hitDatas[RayRadiance].data.photonMapStartIdxs = (int*)photonMapStartIdxs.device;
				rt_hitDatas[ShadowRay].data.photonMapStartIdxs = (int*)photonMapStartIdxs.device;
				rt_hitDatas[RayRadiance].data.NOLT = (int*)NOLT.device;
				rt_hitDatas[ShadowRay].data.NOLT = (int*)NOLT.device;
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
			trans({ {60},{0.01,0.9,0.005},{0.006},{0,0,0},1400.0 }),
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
			{1080,1200},
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
#include <OptiX/_Define_7_Device.h>
#define DefineDevice
#include "Define.h"
#include "device_functions.h"

__global__ void initRandom(curandState* state, unsigned int seed, unsigned int MaxNum)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < MaxNum)
		curand_init(seed, id, 0, state + id);
}
void initRandom(curandState* state, int seed, unsigned int block, unsigned int grid, unsigned int MaxNum)
{
	initRandom << <grid, block >> > (state, seed, MaxNum);
}

__constant__ int NOLT[9];

extern "C" __global__ void GatherKernel(CameraRayData* cameraRayDatas, Photon* photonMap,
	float3* normals, float3* kds, int* photonMapStartIdxs, Parameters& paras)
{
	unsigned int index(blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x);
	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

#ifdef USE_CONNECTRAY
	if (paras.eye == RightEye && (paras.c_image[index].x != 0 || paras.c_image[index].y != 0 || paras.c_image[index].z != 0))
		return;
#endif

	float3 hitPointPosition = cameraRayDatas[index].position;
	float3 hitPointDirection = cameraRayDatas[index].direction;
	int primIdx = cameraRayDatas[index].primIdx;
	float3 normal = normals[primIdx];
	float3 kd = kds[primIdx];

	__shared__ int hashValues[BLOCK_SIZE2];
	__shared__ Photon photons[BLOCK_SIZE2];
	__shared__ int flag;

	flag = 0;

	if (primIdx != -1)
		hashValues[tid] = hash(hitPointPosition);
	else
		hashValues[tid] = -1;

	__syncthreads();

	if (hashValues[tid] != hashValues[(tid + 1) % BLOCK_SIZE2])
		flag = 1;

	__syncthreads();

	if (primIdx == -1)
	{
		paras.image[index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		return;
	}
	
#ifdef USE_SHARED_MEMORY
	if (flag != 0)	// at leasy one thread hit a different hash box 
#endif
	{
		int hitPointHashValue = hash(hitPointPosition);

		float3 indirectFlux = make_float3(0.0f, 1.0f, 0.0f);

		for (int c0(0); c0 < 9; c0++)
		{
			int gridNumber = hitPointHashValue + NOLT[c0];
			int startIdx = photonMapStartIdxs[gridNumber];
			int endIdx = photonMapStartIdxs[gridNumber + 3];
			for (int c1(startIdx); c1 < endIdx; c1++)
			{
				const Photon& photon = photonMap[c1];
				float3 diff = hitPointPosition - photon.position;
				float distance2 = dot(diff, diff);

				if (distance2 <= COLLECT_RAIDUS * COLLECT_RAIDUS && fabsf(dot(diff,normal)) < 0.0001f)
				{
					float Wpc = 1.0f - sqrtf(distance2) / COLLECT_RAIDUS;
					indirectFlux += photon.energy * kd * Wpc;
				}
			}
		}

		indirectFlux /= M_PIf * COLLECT_RAIDUS * COLLECT_RAIDUS * (1.0f - 0.6667f / 1.0f) * PT_PHOTON_CNT;

		paras.image[index] = make_float4(indirectFlux, 1.0f);

#ifdef USE_CONNECTRAY
		if (paras.eye == LeftEye && paras.c_index[index] != -1)
			paras.c_image[paras.c_index[index]] = indirectFlux;
#endif
		//paras.image[index] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
	}
#ifdef USE_SHARED_MEMORY
	else	// all thread hit the same hash box
	{
		float3 indirectFlux = make_float3(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < 9; i++)
		{
			__shared__ Photon* photonStartIdx;
			__shared__ int photonCnt;
			__shared__ int paddedCnt;
			int collectPhotonCnt = 0;

			if (tid == 0)
			{
				int gridNumber = hash(hitPointPosition) + NOLT[i];
				int startIdx = photonMapStartIdxs[gridNumber];
				int endIdx = photonMapStartIdxs[gridNumber + 3];

				photonStartIdx = photonMap + startIdx;
				photonCnt = endIdx - startIdx;
				paddedCnt = ((photonCnt - 1) / BLOCK_SIZE2 + 1) * BLOCK_SIZE2;
			}

			__syncthreads();

			for (int j = tid; j < paddedCnt; j += BLOCK_SIZE2)
			{
				if (j < photonCnt)
					photons[tid] = *(photonStartIdx + j);
				
				__syncthreads();

				for (int k = 0; k < BLOCK_SIZE2 && collectPhotonCnt < photonCnt; k++, collectPhotonCnt++)
				{
					const Photon& photon = photons[k];
					float3 diff = hitPointPosition - photon.position;
					float distance2 = dot(diff, diff);

					if (distance2 <= COLLECT_RAIDUS * COLLECT_RAIDUS && fabsf(dot(diff,normal)) < 0.0001f)
					{
						float Wpc = 1.0f - sqrtf(distance2) / COLLECT_RAIDUS;
						indirectFlux += photon.energy * kd * Wpc;
					}
				}

				__syncthreads();
			}	
		}

		indirectFlux /= M_PIf * COLLECT_RAIDUS * COLLECT_RAIDUS * (1.0f - 0.6667f / 1.0f) * PT_PHOTON_CNT;

		paras.image[index] = make_float4(indirectFlux, 1.0f);
		//paras.image[index] = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
	}
#endif
}

void initNOLT(int* NOLT_host)
{
	cudaMemcpyToSymbol(NOLT, NOLT_host, sizeof(NOLT));
}

void Gather(CameraRayData* cameraRayDatas, Photon* photonMap,
	float3* normals, float3* kds, int* photonMapStartIdxs, uint2 size, Parameters& paras)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(size.x / dimBlock.x, size.y / dimBlock.y);
	GatherKernel << <dimGrid, dimBlock >> > (cameraRayDatas, photonMap,
		normals, kds, photonMapStartIdxs, paras);
}
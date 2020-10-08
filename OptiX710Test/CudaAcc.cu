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

__constant__ int NOLT[27];

extern "C" __global__ void GatherKernel(CameraRayData* cameraRayDatas, Photon* photonMap,
	float3* normals, float3* kds, int* photonMapStartIdxs, Parameters& paras)
{
	unsigned int index(blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x);
	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

	float3 hitPointPosition = cameraRayDatas[index].position;
	float3 hitPointDirection = cameraRayDatas[index].direction;
	int primIdx = cameraRayDatas[index].primIdx;
	float3 kd = kds[primIdx];

	__shared__ int hashValues[BLOCK_SIZE2];
	__shared__ int photonCnt;
	__shared__ Photon photons[1000];
	__shared__ int flag;

	if (tid == 0)
	{
		flag = 0;
		photonCnt = 0;
	}
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
	}
	else if (flag != 0)	// at leasy one thread hit a different hash box 
	{
		int hitPointHashValue = hash(hitPointPosition);

		float3 indirectFlux = make_float3(0.0f, 0.0f, 0.0f);

		for (int c0(0); c0 < 27; c0++)
		{
			int gridNumber = hitPointHashValue + NOLT[c0];
			int startIdx = photonMapStartIdxs[gridNumber];
			int endIdx = photonMapStartIdxs[gridNumber + 1];
			for (int c1(startIdx); c1 < endIdx; c1++)
			{
				const Photon& photon = photonMap[c1];
				float3 diff = hitPointPosition - photon.position;
				float distance = sqrtf(dot(diff, diff));

				if (distance <= COLLECT_RAIDUS)
				{
					float Wpc = 1.0f - distance / COLLECT_RAIDUS;
					indirectFlux += photon.energy * kd * Wpc;
				}
			}
		}

		indirectFlux /= M_PIf * COLLECT_RAIDUS * COLLECT_RAIDUS * (1.0f - 0.6667f / 1.0f) * PT_PHOTON_CNT;

		paras.image[index] = make_float4(indirectFlux, 1.0f);

		//paras.image[index] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
	}
	else	// all thread hit the same hash box
	{
		int hitPointHashValue = hash(hitPointPosition);

		__shared__ int startIdx[27];
		__shared__ int photonCnt[27];
		__shared__ int prefixSum[27];
		__shared__ int totalCnt;

		for (int c0(tid); c0 < 27; c0 += BLOCK_SIZE2)
		{
			int gridNumber = hitPointHashValue + NOLT[c0];
			startIdx[c0] = photonMapStartIdxs[gridNumber];
			photonCnt[c0] = photonMapStartIdxs[gridNumber + 1] - startIdx[c0];
		}

		__syncthreads();

		if (tid == 0)
		{
			prefixSum[0] = 0;
			for (int c0(0); c0 < 26; c0++)
				prefixSum[c0 + 1] = prefixSum[c0] + photonCnt[c0];
			totalCnt = prefixSum[26] + photonCnt[26];
		}

		__syncthreads();

		for (int c0(0); c0 < 27; c0++)
		{
			for (int c1(tid); c1 < photonCnt[c0]; c1 += BLOCK_SIZE2)
			{
				photons[prefixSum[c0] + c1] = *(photonMap + startIdx[c0] + c1);
			}
		}

		__syncthreads();

		float3 indirectFlux = make_float3(0.0f, 0.0f, 0.0f);

		for (int c0(0); c0 < totalCnt; c0++)
		{
			const Photon& photon = photons[c0];
			float3 diff = hitPointPosition - photon.position;
			float distance = sqrtf(dot(diff, diff));

			if (distance <= COLLECT_RAIDUS)
			{
				float Wpc = 1.0f - distance / COLLECT_RAIDUS;
				indirectFlux += photon.energy * kd * Wpc;
			}
		}

		indirectFlux /= M_PIf * COLLECT_RAIDUS * COLLECT_RAIDUS * (1.0f - 0.6667f / 1.0f) * PT_PHOTON_CNT;

		paras.image[index] = make_float4(indirectFlux, 1.0f);
		//paras.image[index] = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
	}
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
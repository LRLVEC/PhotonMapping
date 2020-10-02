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

struct PhotonMaxHeap
{
#define PHOTONHEAP_SIZE 127
	int currentSize;
	HeapPhoton photons[PHOTONHEAP_SIZE];
	float3 cameraRayDir;

#define PARENT(x) ((x - 1) >> 1)
	__device__ __inline__ void siftUp(int position)
	{
		int tempPos = position;
		HeapPhoton tempPhoton = photons[tempPos];
		while (tempPos > 0 && photons[PARENT(tempPos)].distance2 < tempPhoton.distance2)
		{
			photons[tempPos] = photons[PARENT(tempPos)];
			tempPos = PARENT(tempPos);
		}
		photons[tempPos] = tempPhoton;
	}

	__device__ __inline__ void siftDown(int position)
	{
		int i = position;
		int j = 2 * i + 1;
		HeapPhoton temp = photons[i];
		while (j < currentSize)
		{
			if ((j < currentSize - 1) && photons[j].distance2 < photons[j + 1].distance2)
				j++;
			if (temp.distance2 < photons[j].distance2)
			{
				photons[i] = photons[j];
				i = j;
				j = 2 * j + 1;
			}
			else break;
		}
		photons[i] = temp;
	}

	__device__ __inline__ void push(const float& distance2, const int& index)
	{
		if (currentSize < PHOTONHEAP_SIZE)
		{
			photons[currentSize].distance2 = distance2;
			photons[currentSize].index = index;
			siftUp(currentSize);
			currentSize++;
			return;
		}
		else if (photons[0].distance2 > distance2)
		{
			photons[0].distance2 = distance2;
			photons[0].index = index;
			siftDown(0);
		}
	}

#define filter_k 1.1

	__device__ __inline__ float3 accumulate(float& radius2, const float3 * kds, const float3 * normals, const Photon * sharedPhotons)
	{
		float3 flux = make_float3(0.0f, 0.0f, 0.0f);
		float Wpc = 0.0f;	// weight of cone filter

		float radius = sqrtf(radius2);

		for (int c0(0); c0 < currentSize; c0++)
		{
			const Photon& photon = sharedPhotons[photons[c0].index];
			float3 hitPointNormal = normals[photon.primIdx];
			if (dot(photon.dir, hitPointNormal) * dot(cameraRayDir, hitPointNormal) <= 0)
				continue;
			Wpc = 1.0f - sqrtf(photons[c0].distance2) / (filter_k * radius);
			flux += photon.energy * kds[photon.primIdx] * Wpc;
		}

		flux = flux / (M_PIf * radius2) / (1 - 0.6667f / filter_k) / PT_PHOTON_CNT;

		return flux;
	}
};

__constant__ int NOLT[27];

extern "C" __global__ void GatherKernel(CameraRayData* cameraRayDatas, Photon* photonMap,
	float3* normals, float3* kds, int* photonMapStartIdxs, Parameters& paras)
{
	unsigned int index(blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x);
	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

	float3 hitPointPosition = cameraRayDatas[index].position;
	float3 hitPointDirection = cameraRayDatas[index].direction;
	int primIdx = cameraRayDatas[index].primIdx;

	__shared__ int hashValues[BLOCK_SIZE2];
	__shared__ int photonCnt;
	__shared__ Photon photons[800];
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
		//atomicAdd(&flag, 1);
		flag = 1;

	__syncthreads();

	if (primIdx == -1)
	{
		paras.image[index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	}
	else if (flag != 0)	// at leasy one thread hit a different hash box 
	{
		float radius2 = COLLECT_RAIDUS * COLLECT_RAIDUS;

		int hitPointHashValue = hash(hitPointPosition);

		PhotonMaxHeap heap;
		heap.currentSize = 0;
		heap.cameraRayDir = hitPointDirection;

		for (int c0(0); c0 < 27; c0++)
		{
			int gridNumber = hitPointHashValue + NOLT[c0];
			int startIdx = photonMapStartIdxs[gridNumber];
			int endIdx = photonMapStartIdxs[gridNumber + 1];
			for (int c1(startIdx); c1 < endIdx; c1++)
			{
				Photon* photon = photonMap + c1;
				float3 diff = hitPointPosition - photon->position;
				float distance2 = dot(diff, diff);

				if (distance2 <= radius2)
					heap.push(distance2, c1);
			}
		}

		float3 indirectFlux = heap.accumulate(radius2, kds, normals, photonMap);

		paras.image[index] = make_float4(indirectFlux, 1.0f);

		//paras.image[index] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
	}
	else	// all thread hit the same hash box
	{
		int hitPointHashValue = hash(hitPointPosition);

		__shared__ int gridNumber;
		__shared__ int startIdx;
		__shared__ int endIdx;

		for (int c0(0); c0 < 27; c0++)
		{
			if (tid == 0)
			{
				gridNumber = hitPointHashValue + NOLT[c0];
				startIdx = photonMapStartIdxs[gridNumber];
				endIdx = photonMapStartIdxs[gridNumber + 1];
			}
			
			__syncthreads();

			for (int c1(startIdx + tid); c1 < endIdx; c1 += BLOCK_SIZE2)
			{

				photons[photonCnt + c1 - startIdx] = *(photonMap + c1);
			}

			__syncthreads();

			if (tid == 0)
			{
				photonCnt += endIdx - startIdx;
			}
		}

		__syncthreads();

		float radius2 = COLLECT_RAIDUS * COLLECT_RAIDUS;
		PhotonMaxHeap heap;
		heap.currentSize = 0;
		heap.cameraRayDir = hitPointDirection;

		for (int c0(0); c0 < photonCnt; c0++)
		{
			float3 diff = hitPointPosition - photons[c0].position;
			float distance2 = dot(diff, diff);

			if (distance2 <= radius2)
				heap.push(distance2, c0);
		}

		float3 indirectFlux = heap.accumulate(radius2, kds, normals, photons);

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
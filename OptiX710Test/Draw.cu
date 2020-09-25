#define DefineDevice
#include "Define.h"

extern "C"
{
	__constant__ Parameters paras;
}


struct PhotonMaxHeap
{
#define PHOTONHEAP_SIZE 31
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

	__device__ __inline__ float3 accumulate(float& radius2)
	{
		Rt_HitData* hitData = (Rt_HitData*)optixGetSbtDataPointer();
		float3* kds = hitData->kds;
		float3* normals = hitData->normals;
		Photon* photonMap = hitData->photonMap;

		float3 flux = make_float3(0.0f, 0.0f, 0.0f);
		float Wpc = 0.0f;	// weight of cone filter

		if (currentSize > 0)
			radius2 = fmaxf(photons[0].distance2, COLLECT_RAIDUS * COLLECT_RAIDUS);
		float radius = sqrtf(radius2);

		for (int c0(0); c0 < currentSize; c0++)
		{
			const Photon& photon = photonMap[photons[c0].index];
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


extern "C" __global__ void __raygen__RayAllocator()
{
	uint2 index = make_uint2(optixGetLaunchIndex());

	float2 ahh = random(index, paras.size, 0) +
		make_float2(index) - make_float2(paras.size) / 2.0f;
	float4 d = make_float4(ahh, paras.trans->z0, 0);
	float3 dd = normalize(make_float3(
		dot(paras.trans->row0, d),
		dot(paras.trans->row1, d),
		dot(paras.trans->row2, d)));

	optixTrace(paras.handle, paras.trans->r0, dd,
		0.001f, 1e16f,
		0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance);		// missSBTIndex
}


extern "C" __global__ void __closesthit__RayRadiance()
{
	int primIdx = optixGetPrimitiveIndex();
	uint2 index = make_uint2(optixGetLaunchIndex());

	Rt_HitData* hitData = (Rt_HitData*)optixGetSbtDataPointer();
	float3 rayDir = optixGetWorldRayDirection();

	// hit point info
	float3 hitPointPosition = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;
	float3 hitPointKd = hitData->kds[primIdx];
	float3 hitPointNormal = hitData->normals[primIdx];

	if (fmaxf(hitPointKd) > 0.0f)
	{
		Photon* photonMap = hitData->photonMap;
		int* photonMapStartIdxs = hitData->photonMapStartIdxs;
		int* NOLT = hitData->NOLT;

		float radius2 = COLLECT_RAIDUS * COLLECT_RAIDUS;

		int hitPointHashValue = hash(hitPointPosition);

		PhotonMaxHeap heap;
		heap.currentSize = 0;
		heap.cameraRayDir = rayDir;

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

		//DebugData& debug = hitData->debugDatas[index.y * paras.size.x + index.x];

		// indirect flux
		float3 indirectFlux = heap.accumulate(radius2);

		// direct flux
		float3 lightSourcePosition;
		LightSource* lightSource = hitData->lightSource;
		
		lightSourcePosition = lightSource->position;

		float3 shadowRayDir = lightSourcePosition - hitPointPosition;
		float Tmax = sqrtf(dot(shadowRayDir, shadowRayDir));
		shadowRayDir = normalize(shadowRayDir);
		float cosDN = fabsf(dot(shadowRayDir, hitPointNormal));// NOTE: fabsf?

		float3 directFlux = make_float3(0.0f, 0.0f, 0.0f);

		if (dot(shadowRayDir, hitPointNormal) * dot(rayDir, hitPointNormal) < 0)
		{
			float attenuation = 1.0f;
			unsigned int pd0, pd1;
			pP(&attenuation, pd0, pd1);
			unsigned int pd2, pd3;
			pP(&Tmax, pd2, pd3);
			optixTrace(paras.handle, hitPointPosition, shadowRayDir,
				0.001f, 1e16f,
				0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
				ShadowRay,        // SBT offset
				RayCount,           // SBT stride
				ShadowRay,        // missSBTIndex
				pd0, pd1, pd2, pd3);

			directFlux = lightSource->power * attenuation * hitPointKd * cosDN;
		}

		//float3 color = 0.1f * directFlux;
		float3 color = indirectFlux;

		paras.image[index.y * paras.size.x + index.x] = make_float4(color, 1.0f);
	}
}

extern "C" __global__ void __closesthit__ShadowRay()
{
	unsigned int pd0, pd1;
	unsigned int pd2, pd3;
	pd0 = optixGetPayload_0();
	pd1 = optixGetPayload_1();
	pd2 = optixGetPayload_2();
	pd3 = optixGetPayload_3();
	float Tmax = *(float*)uP(pd2, pd3);
	if (Tmax > optixGetRayTmax())
		* (float*)uP(pd0, pd1) = 0.0f;
}

extern "C" __global__ void __miss__RayRadiance()
{
	uint2 index = make_uint2(optixGetLaunchIndex());
	paras.image[index.y * paras.size.x + index.x] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}

extern "C" __global__ void __miss__ShadowRay()
{
	
}

// create a orthonormal basis from normalized vector n
static __device__ __inline__ void createOnb(const float3& n, float3& U, float3& V)
{
	U = cross(n, make_float3(0.0f, 1.0f, 0.0f));
	if (dot(U, U) < 1e-3)
		U = cross(n, make_float3(1.0f, 0.0f, 0.0f));
	U = normalize(U);
	V = cross(n, U);
}

extern "C" __global__ void __raygen__PhotonEmit()
{
	unsigned int index = optixGetLaunchIndex().x;
	int startIdx = index * PT_MAX_DEPOSIT;

	curandState* statePtr = paras.randState + index;
	curandStateMini state;
	getCurandState(&state, statePtr);

	Pt_RayGenData* raygenData = (Pt_RayGenData*)optixGetSbtDataPointer();
	LightSource* lightSource = raygenData->lightSource;	

	// cos sampling(should be uniformly sampling a sphere)
	float3 dir = randomDirectionCosN(normalize(lightSource->direction), 1.0f, &state);
	float3 position = lightSource->position;

	/*float2 seed = curand_uniform2(&state);
	float z = 1 - 2 * seed.x;
	float r = sqrtf(fmax(0.0f, 1.0f - z * z));
	float phi = 2 * M_PIf * seed.y;
	float3 scale = make_float3(r * cosf(phi), r * sinf(phi), z);
	float3 lightDir = normalize(lightSource->direction);
	float3 U, V;
	createOnb(lightDir, U, V);

	position = lightSource->position;
	dir = normalize(lightDir * scale.z + U * scale.x + V * scale.y);*/

	// initialize photon records
	Photon* photons = raygenData->photons;
	for (int c0(0); c0 < PT_MAX_DEPOSIT; c0++)
		photons[startIdx + c0].energy = make_float3(0.0f);

	// set ray payload
	PhotonPrd prd;
	prd.energy = lightSource->power;
	prd.startIdx = startIdx;
	prd.numDeposits = 0;
	prd.depth = 0;
	unsigned int pd0, pd1;
	pP(&prd, pd0, pd1);

	// trace the photon
	optixTrace(paras.handle, position, dir,
		0.001f, 1e16f,
		0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance,        // missSBTIndex
		pd0, pd1, state.d, state.v[0], state.v[1], state.v[2], state.v[3], state.v[4]);
	setCurandState(statePtr, &state);
}

extern "C" __global__ void __closesthit__PhotonHit()
{
	uint2 index = make_uint2(optixGetLaunchIndex());
	Pt_HitData* hitData = (Pt_HitData*)optixGetSbtDataPointer();
	int primIdx = optixGetPrimitiveIndex();

	curandStateMini state(getCurandStateFromPayload());

	// calculate the hit point
	float3 hitPointPosition = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
	float3 hitPointNormal = hitData->normals[primIdx];

	unsigned int pd0, pd1;
	pd0 = optixGetPayload_0();
	pd1 = optixGetPayload_1();
	PhotonPrd prd = *(PhotonPrd*)uP(pd0, pd1);

	float3 newDir;
	float3 oldDir = optixGetWorldRayDirection();

	float3 kd = hitData->kds[primIdx];
	if (fmaxf(kd) > 0.0f)
	{
		// hit a diffuse surface; record hit if it has bounced at least once
		// NOTE: depth > 0 or >= 0?
		if (prd.depth >= 0)
		{
			Photon& photon = hitData->photons[prd.startIdx + prd.numDeposits];
			photon.position = hitPointPosition;
			photon.dir = oldDir;
			photon.energy = prd.energy;
			photon.primIdx = primIdx;
			prd.numDeposits++;
		}

		// Russian roulette
		float Pd = fmaxf(kd);	// probability of being diffused
		if (curand_uniform(&state) > Pd)
			return;	// absorb

		prd.energy = kd * prd.energy / Pd;

		// cosine-weighted hemisphere sampling
		float3 W = { 0.f,0.f,0.f };
		if (dot(oldDir, hitPointNormal) > 0)
			W = -normalize(hitPointNormal);
		else
			W = normalize(hitPointNormal);

		newDir = randomDirectionCosN(W, 1.0f, &state);
	}

	prd.depth++;

	if (prd.numDeposits >= PT_MAX_DEPOSIT || prd.depth >= PT_MAX_DEPTH)
		return;

	pP(&prd, pd0, pd1);

	optixTrace(paras.handle, hitPointPosition, newDir,
		0.001f, 1e16f,
		0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance,        // missSBTIndex
		pd0, pd1, state.d, state.v[0], state.v[1], state.v[2], state.v[3], state.v[4]);
}

extern "C" __global__ void __miss__Ahh()
{

}
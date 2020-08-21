#define DefineDevice
#include "Define.h"

extern "C"
{
	__constant__ Parameters paras;
}

extern "C" __global__ void __raygen__RayAllocator()
{
	uint2 index = make_uint2(optixGetLaunchIndex());

	Rt_RayGenData* raygenData = (Rt_RayGenData*)optixGetSbtDataPointer();
	CameraRayHitData& cameraRayHitData = raygenData->cameraRayHitDatas[index.y * paras.size.x + index.x];

	// initialize the camera ray hit data
	cameraRayHitData.position = make_float3(0.0f);
	cameraRayHitData.rayDir = make_float3(0.0f);
	cameraRayHitData.primIdx = -1;

	float3 color = make_float3(0.f, 0.f, 0.f);

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

extern "C" __global__ void __closesthit__RayHit()
{
	int primIdx = optixGetPrimitiveIndex();
	uint2 index = make_uint2(optixGetLaunchIndex());

	Rt_CloseHitData* closeHitData = (Rt_CloseHitData*)optixGetSbtDataPointer();
	CameraRayHitData& cameraRayHitData = closeHitData->cameraRayHitDatas[index.y * paras.size.x + index.x];

	//calculate the hit point
	float3 hitPoint = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

	// record the information of the hitpoint
	float3 kd = closeHitData->kds[primIdx];
	if (fmaxf(kd) > 0.0f)
	{
		// hit a diffuse surface
		cameraRayHitData.position = hitPoint;
		cameraRayHitData.rayDir = optixGetWorldRayDirection();
		cameraRayHitData.primIdx = primIdx;
	}
	else
	{
		// hit a specular surface
	}
}
extern "C" __global__ void __miss__Ahh()
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
	uint2 index = make_uint2(optixGetLaunchIndex());
	int startIdx = (index.y * paras.pt_size.x + index.x) * paras.maxPhotonCnt;

	float3 position = make_float3(0.0f);
	float3 dir = make_float3(0.0f);

	Pt_RayGenData* raygenData = (Pt_RayGenData*)optixGetSbtDataPointer();
	LightSource* lightSource = raygenData->lightSource;	

	if (lightSource->type == LightSource::SPOT)
	{
		float2 seed = raygenData->directionSeeds[index.y * paras.pt_size.x + index.x];
		// uniformly sampling a sphere
		float z = 1 - 2 * seed.x;
		float r = sqrtf(fmax(0.0f, 1.0f - z * z));
		float phi = 2 * M_PIf * seed.y;
		float3 scale = make_float3(r * cosf(phi), r * sinf(phi), z);
		float3 lightDir = normalize(lightSource->direction);
		float3 U, V;
		createOnb(lightDir, U, V);

		position = lightSource->position;
		dir = normalize(lightDir * scale.z + U * scale.x + V * scale.y);
	}
	else if (lightSource->type == LightSource::SQUARE)
	{
		float2 positionSeed = raygenData->positionSeeds[index.y * paras.pt_size.x + index.x];
		float2 directionSeed = raygenData->directionSeeds[index.y * paras.pt_size.x + index.x];
		// cosine-weighted hemisphere sampling
		float3 U, V, W;
		W = normalize(lightSource->direction);
		createOnb(W, U, V);
		float2 scale = randomCircle(directionSeed);
		float z = 1.0f - dot(scale, scale);
		z = z > 0.0f ? sqrtf(z) : 0.0f;

		position = lightSource->position + lightSource->edge1 * positionSeed.x + lightSource->edge2 * positionSeed.y;
		dir = normalize(W * z + U * scale.x + V * scale.y);
	}

	// initialize photon records
	Photon* photons = raygenData->photons;
	for (int c0(0); c0 < paras.maxPhotonCnt; c0++)
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
		pd0, pd1);
}

extern "C" __global__ void __closesthit__PhotonHit()
{
	uint2 index = make_uint2(optixGetLaunchIndex());
	Pt_CloseHitData* closeHitData = (Pt_CloseHitData*)optixGetSbtDataPointer();
	int primIdx = optixGetPrimitiveIndex();
	float2 seed = closeHitData->directionSeeds[index.y * paras.pt_size.x + index.x];

	// calculate the hit point
	float3 hitPointPosition = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
	float3 hitPointNormal = closeHitData->normals[primIdx];

	unsigned int pd0, pd1;
	pd0 = optixGetPayload_0();
	pd1 = optixGetPayload_1();
	PhotonPrd prd = *(PhotonPrd*)uP(pd0, pd1);

	float3 newDir;
	float3 oldDir = optixGetWorldRayDirection();

	// get random seed for Russian roulette
	float RRSeed = closeHitData->RRseeds[(index.y * paras.pt_size.x + index.x) * paras.maxDepth + prd.depth];

	float3 kd = closeHitData->kds[primIdx];
	if (fmaxf(kd) > 0.0f)
	{
		// hit a diffuse surface; record hit if it has bounced at least once
		// NOTE: depth > 0 or >= 0?
		if (prd.depth >= 0)
		{
			Photon& photon = closeHitData->photons[prd.startIdx + prd.numDeposits];
			photon.position = hitPointPosition;
			photon.normal = hitPointNormal;
			photon.dir = oldDir;
			photon.energy = prd.energy;
			prd.numDeposits++;
		}

		float Pd = fmaxf(kd);	// probability of being absorbed
		// Russian roulette : absorb
		if (RRSeed > Pd)
			return;

		prd.energy = kd * prd.energy / Pd;

		// cosine-weighted hemisphere sampling
		float3 U, V, W;
		if (dot(oldDir, hitPointNormal) > 0)
			W = -normalize(hitPointNormal);
		else
			W = normalize(hitPointNormal);
		createOnb(W, U, V);
		float2 scale = randomCircle(seed);
		float z = 1.0f - dot(scale, scale);
		z = z > 0.0f ? sqrtf(z) : 0.0f;

		newDir = normalize(W * z + U * scale.x + V * scale.y);
	}

	prd.depth++;

	if (prd.numDeposits >= paras.maxPhotonCnt || prd.depth >= paras.maxDepth)
		return;

	pP(&prd, pd0, pd1);

	optixTrace(paras.handle, hitPointPosition, newDir,
		0.001f, 1e16f,
		0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance,        // missSBTIndex
		pd0, pd1);
}

// used in KNN photon search
struct HeapPhoton
{
	float distance2;
	float3 flux;
	float3 kd;
};

struct PhotonMaxHeap
{
#define PHOTONHEAP_SIZE 31
	int currentSize;
	HeapPhoton photons[PHOTONHEAP_SIZE];

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

	__device__ __inline__ void push(const float& distance2, const float3& flux, const float3& kd)
	{
		if (currentSize < PHOTONHEAP_SIZE)
		{
			photons[currentSize].distance2 = distance2;
			photons[currentSize].flux = flux;
			photons[currentSize].kd = kd;
			siftUp(currentSize);
			currentSize++;
			return;
		}
		else if (photons[0].distance2 > distance2)
		{
			photons[0].distance2 = distance2;
			photons[0].flux = flux;
			photons[0].kd = kd;
			siftDown(0);
		}
	}

	__device__ __inline__ float3 accumulate(float& radius2)
	{
		float3 flux = make_float3(0.0f, 0.0f, 0.0f);
		for (int c0(0); c0 < currentSize; c0++)
			flux += photons[c0].flux * photons[c0].kd;
		if (currentSize > 0)
			radius2 = photons[0].distance2;
		return flux;
	}
};

#define MAX_DEPTH 20	// one million photons

extern "C" __global__ void __raygen__Gather()
{
	uint2 index = make_uint2(optixGetLaunchIndex());

	Gt_RayGenData* raygenData = (Gt_RayGenData*)optixGetSbtDataPointer();
	const CameraRayHitData& cameraRayHitData = raygenData->cameraRayHitDatas[index.y * paras.size.x + index.x];
	Photon* photonMap = raygenData->photonMap;

	int primIdx = cameraRayHitData.primIdx;

	if (primIdx == -1)
	{
		paras.image[index.y * paras.size.x + index.x] = make_float4(0.f, 0.f, 0.f, 1.0f);
		return;
	}

	float3 hitPointPosition = cameraRayHitData.position;
	float3 hitPointNormal = raygenData->normals[primIdx];
	float3 hitPointKd = raygenData->kds[primIdx];

	float radius2 = 0.0001;

	int stack[MAX_DEPTH];
	int stackTop = 0;
	int node = 0;

#define push_node(N) stack[stackTop++] = (N)
#define pop_node() stack[--stackTop]

	PhotonMaxHeap heap;
	heap.currentSize = 0;

	push_node(0);
	do
	{
		Photon& photon = photonMap[node];
		if (!(photon.axis & PPM_NULL))
		{
			float3 diff = hitPointPosition - photon.position;
			float distance2 = dot(diff, diff);

			if (distance2 <= radius2)
				heap.push(distance2, photon.energy, hitPointKd);

			if (!(photon.axis & PPM_LEAF))
			{
				float d;
				if (photon.axis & PPM_X)
					d = diff.x;
				else if (photon.axis & PPM_Y)
					d = diff.y;
				else
					d = diff.z;

				int selector = d < 0.0f ? 0 : 1;
				if (d * d < radius2)
					push_node(2 * node + 2 - selector);

				node = 2 * node + 1 + selector;
			}
			else
				node = pop_node();
		}
		else
			node = pop_node();
	} while (node);

	// indirect flux
	float3 flux = heap.accumulate(radius2);
	float3 indirectFlux = 1.0f / (M_PIf * radius2) * flux / (paras.pt_size.x * paras.pt_size.y);

	// direct flux
	float3 lightSourcePosition;
	LightSource* lightSource = raygenData->lightSource;
	if (lightSource->type == LightSource::SPOT)
	{
		lightSourcePosition = lightSource->position;
	}
	else if (lightSource->type == LightSource::SQUARE)
	{
		// To do: 如何计算面光源
		lightSourcePosition = lightSource->position;
	}

	float3 shadowRayDir = lightSourcePosition - hitPointPosition;
	float Tmax = sqrtf(dot(shadowRayDir, shadowRayDir));
	shadowRayDir = normalize(shadowRayDir);
	float cosDN = fabsf(dot(shadowRayDir, hitPointNormal));// NOTE: fabsf?

	float attenuation = 1.0f;
	unsigned int pd0, pd1;
	pP(&attenuation, pd0, pd1);
	unsigned int pd2, pd3;
	pP(&Tmax, pd2, pd3);
	optixTrace(paras.handle, hitPointPosition, shadowRayDir,
		0.001f, 1e16f,
		0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance,        // missSBTIndex
		pd0, pd1, pd2, pd3);

	float3 directFlux = lightSource->power * attenuation * hitPointKd * cosDN;

	//float3 color = directFlux;
	//float3 color = indirectFlux;
	float3 color = directFlux + indirectFlux;

	paras.image[index.y * paras.size.x + index.x] = make_float4(color, 1.0f);
}

extern "C" __global__ void __closesthit__ShadowRayHit()
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
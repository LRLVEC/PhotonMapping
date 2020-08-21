#define DefineDevice
#include "Define.h"


//To do: Construct again...

extern "C"
{
	__constant__ Parameters paras;
}

extern "C" __global__ void __raygen__RayAllocator()
{
	uint2 index = make_uint2(optixGetLaunchIndex());
	Rt_RayGenData* rt_raygenData = (Rt_RayGenData*)optixGetSbtDataPointer();
	CameraRayHitData& cameraRayHitData = rt_raygenData->cameraRayHitData[index.y * paras.size.x + index.x];

	// initialize the camera ray hit data
	cameraRayHitData.position = make_float3(0.0f);
	cameraRayHitData.rayDir = make_float3(0.0f);
	cameraRayHitData.primIdx = 0;
	cameraRayHitData.hit = false;

	// generate the camera ray direction
	float2 ahh = random(index, paras.size, 0) +
		make_float2(index) - make_float2(paras.size) / 2.0f;
	float4 d = make_float4(ahh, paras.trans->z0, 0);
	float3 dd = normalize(make_float3(
		dot(paras.trans->row0, d),
		dot(paras.trans->row1, d),
		dot(paras.trans->row2, d)));

	optixTrace(paras.handle, paras.trans->r0, dd,
		0.001f, 1e16f,
		0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance);       // missSBTIndex
}


extern "C" __global__ void __closesthit__RayHit()
{
	int primIdx = optixGetPrimitiveIndex();
	uint2 index = make_uint2(optixGetLaunchIndex());

	Rt_CloseHitData* closeHitData = (Rt_CloseHitData*)optixGetSbtDataPointer();
	CameraRayHitData& cameraRayHitData = closeHitData->cameraRayHitData[index.y * paras.size.x + index.x];

	// calculate the hit point: right?
	float3 hitPoint = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

	// record the information of the hitpoint
	float3 kd = closeHitData->kd[primIdx];
	if (fmaxf(kd) > 0.0f)
	{
		// hit a diffuse surface
		cameraRayHitData.position = hitPoint;
		cameraRayHitData.rayDir = optixGetWorldRayDirection();
		cameraRayHitData.primIdx = primIdx;
		cameraRayHitData.hit = true;
	}
	//else
	//{
	//	// hit a specular surface
	//}
}


extern "C" __global__ void __miss__Ahh()
{
}


// create a ortho-normal basis from normalized vector n
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
	int pm_index = (index.y * paras.pt_size.x + index.x) * paras.maxPhotonCnt;

	float3 rayDir = make_float3(0.0f);

	// decide the ray direction
	Pt_RayGenData* pt_raygenData = (Pt_RayGenData*)optixGetSbtDataPointer();
	LightSource* lightSource = pt_raygenData->lightSource;
	if (lightSource->type == LightSource::SPOT)
	{
		// To do: get the direction again with random seed
		float2 u = random(index, paras.pt_size, 0);
		float z = u.x;
		float r = sqrtf(fmax(0.0f, 1.0f - z * z));
		float phi = 2 * M_PIf * u.y;
		float3 scale = make_float3(r * cosf(phi), r * sinf(phi), z);
		float3 lightDir = normalize(lightSource->direction);
		float3 U, V;
		createOnb(lightDir, U, V);
		rayDir = normalize(lightDir * scale.x + U * scale.y + V * scale.z);
	}
	else if (lightSource->type == LightSource::AREA)
	{

	}

	// initialize photon records
	PhotonRecord* photonRecord = pt_raygenData->photonRecord;
	for (int c0(0); c0 < paras.maxPhotonCnt; c0++)
		photonRecord[pm_index + c0].energy = make_float3(0.0f);

	// set ray payload
	PhotonPrd prd;
	prd.energy = lightSource->power;
	prd.pm_index = pm_index;
	prd.numDeposits = 0;
	prd.depth = 0;
	unsigned int pd0, pd1;
	pP(&prd, pd0, pd1);

	// trace the photon
	optixTrace(paras.handle, lightSource->position, rayDir,
		0.001f, 1e16f,
		0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance,        // missSBTIndex
		pd0, pd1);
}

extern "C" __global__ void __closesthit__PhotonHit()
{
	Pt_CloseHitData* closeHitData = (Pt_CloseHitData*)optixGetSbtDataPointer();
	int primIdx = optixGetPrimitiveIndex();

	// calculate the hit point: right?
	float3 hitPoint = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

	unsigned int pd0, pd1;
	pd0 = optixGetPayload_0();
	pd1 = optixGetPayload_1();
	PhotonPrd prd = *(PhotonPrd*)uP(pd0, pd1);

	float3 newRayDir;

	float3 kd = closeHitData->kd[primIdx];
	if (fmaxf(kd) > 0.0f)
	{
		// hit a diffuse surface; record hit if it has bounced at least once
		if (prd.depth > 0)
		{
			PhotonRecord& record = closeHitData->photonRecord[prd.pm_index + prd.numDeposits];
			record.position = hitPoint;
			record.normal = closeHitData->normals[primIdx];
			record.rayDir = optixGetWorldRayDirection();
			record.energy = prd.energy;
			prd.numDeposits++;
		}

		// set the new ray
		prd.energy = kd * prd.energy;	// To do: RR
		float3 U, V, W;
		W = normalize(closeHitData->normals[primIdx]);
		createOnb(W, U, V);
		float2 seed;	// To do: get the seed
		float2 scale = randomCircle(seed);
		float z = 1.0f - dot(scale, scale);
		z = fmaxf(sqrtf(z), 0.0f);

	}
	/*else
	{
		prd.energy = ks * prd.energy;
		newRayDir = ...
	}*/

	prd.depth++;
	if (prd.numDeposits >= paras.maxPhotonCnt || prd.depth >= paras.maxDepth)
		return;

	// set ray payload
	pP(&prd, pd0, pd1);

	optixTrace(paras.handle, hitPoint, newRayDir,
		0.001f, 1e16f,
		0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance,        // missSBTIndex
		pd0, pd1);
}


static __device__ __inline__
void accumulatePhoton(const PhotonRecord& photon,
	const float3& hitPointNormal,
	const float3& hitPointAttenKd,
	int& numNewPhotons, float3& flux)
{
	if (dot(photon.normal, hitPointNormal) > 0.01f)	// ?
	{
		float3 newFlux = photon.energy * hitPointAttenKd;
		flux += newFlux;
		numNewPhotons++;
	}
}


#define MAX_DEPTH 20	// one million photons

extern "C" __global__ void __raygen__Gather()
{
	uint2 index = make_uint2(optixGetLaunchIndex());

	paras.image[index.y * paras.size.x + index.x] = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
	return;

	Gather_RayGenData* gather_raygenData = (Gather_RayGenData*)optixGetSbtDataPointer();
	const CameraRayHitData& cameraRayHitData = gather_raygenData->cameraRayHitData[index.y * paras.size.x + index.x];
	PhotonRecord* photonMap = gather_raygenData->photonMap;

	if (!cameraRayHitData.hit)
	{
		paras.image[index.y * paras.size.x + index.x] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		return;
	}

	// get the property of the hit point
	float3 hitPointNormal = gather_raygenData->normals[cameraRayHitData.primIdx];
	float3 hitPointAttenKd = gather_raygenData->attenKd[cameraRayHitData.primIdx];

	float radius2 = 0.02;	// To do: dynamic adjust

	int stack[MAX_DEPTH];
	int stackPointer = 0;
	int node = 0;

#define push_node(N) stack[stackPointer++] = (N)
#define pop_node(N) stack[--stackPointer]

	int numNewPhotons = 0;
	float3 flux = make_float3(0.0f, 0.0f, 0.0f);

	push_node(0);
	do
	{
		PhotonRecord& photon = photonMap[node];
		if (!(photon.axis & PPM_NULL))
		{
			float3 diff = cameraRayHitData.position - photon.position;
			float distance2 = dot(diff, diff);

			if (distance2 <= radius2)
				accumulatePhoton(photon, hitPointNormal, hitPointAttenKd, numNewPhotons, flux);

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
			{
				node = pop_node();
			}
		}
		else
		{
			node = pop_node();
		}
	} while (node);

	// compute the indirect flux
	float3 indirectFlux = 1.0f / (M_PIf * radius2) * flux / (paras.pt_size.x * paras.pt_size.y);

	// compute the direct flux
	float3 pointOnLight;
	LightSource* lightSource = gather_raygenData->lightSource;
	if (lightSource->type == LightSource::SPOT)
	{
		pointOnLight = lightSource->position;
	}
	/*else if (lightSource->type == LightSource::AREA)
	{

	}*/

	float3 shadowRayDir = normalize(pointOnLight - cameraRayHitData.position);
	float cosDN = dot(shadowRayDir, hitPointNormal);

	float attenuation = 1.0f;
	unsigned int pd0, pd1;
	pP(&attenuation, pd0, pd1);
	optixTrace(paras.handle, cameraRayHitData.position, shadowRayDir,
		0.001f, 1e16f,
		0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance,        // missSBTIndex
		pd0, pd1);

	float3 directFlux = lightSource->power * attenuation * hitPointAttenKd;
	float3 finalColor = indirectFlux + directFlux;
	paras.image[index.y * paras.size.x + index.x] = make_float4(finalColor, 1.0f);
}


extern "C" __global__ void __closesthit__ShadowRayHit()
{
	unsigned int pd0, pd1;
	pd0 = optixGetPayload_0();
	pd1 = optixGetPayload_1();
	*(float*)uP(pd0, pd1) = 0.0f;	// invisiable
	optixTerminateRay();
}
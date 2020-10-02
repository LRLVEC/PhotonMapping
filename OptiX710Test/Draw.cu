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
	CameraRayData& cameraRayData = raygenData->cameraRayDatas[index.y * paras.size.x + index.x];
	cameraRayData.primIdx = -1;

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
		// record cameraRay info
		CameraRayData& cameraRayData = hitData->cameraRayDatas[index.y * paras.size.x + index.x];
		cameraRayData.position = hitPointPosition;
		cameraRayData.direction = rayDir;
		cameraRayData.primIdx = primIdx;

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

		float3 color = 0.1f * directFlux;

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
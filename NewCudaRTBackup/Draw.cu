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
	RayData* rtData = (RayData*)optixGetSbtDataPointer();
	float3 color = make_float3(0);

	float2 ahh = random(index, paras.size, 0) +
		make_float2(index) - make_float2(paras.size) / 2.0f;
	float4 d = make_float4(ahh, paras.trans->z0, 0);
	float3 dd = normalize(make_float3(
		dot(paras.trans->row0, d),
		dot(paras.trans->row1, d),
		dot(paras.trans->row2, d)));
	unsigned int pd0, pd1;
	pP(&color, pd0, pd1);
	optixTrace(paras.handle, paras.trans->r0, dd,
		0.001f, 1e16f,
		0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance,        // missSBTIndex
		pd0, pd1);
	paras.image[index.y * paras.size.x + index.x] = make_float4(color, 1.0f);
}
extern "C" __global__ void __closesthit__Ahh()
{
	CloseHitData* closeHitData = (CloseHitData*)optixGetSbtDataPointer();
	int primIdx = optixGetPrimitiveIndex();
	float3 n = closeHitData->normals[primIdx];
	unsigned int pd0, pd1;
	pd0 = optixGetPayload_0();
	pd1 = optixGetPayload_1();
	*(float3*)uP(pd0, pd1) = normalize(n + 1) / 2 * make_float3(tanh(1.0 / optixGetRayTmax()));
}
extern "C" __global__ void __miss__Ahh()
{
}

#pragma once
#ifdef DefineDevice
#include <OptiX/_Define_7_Device.h>
#else
using TransInfo = OpenGL::OptiX::Trans::TransInfo;
#endif
enum RayType
{
	RayRadiance = 0,
	RayCount
};
struct RayData
{
	float r, g, b;
};
struct CloseHitData
{
	float3* normals;
};
struct Parameters
{
	float4* image;
	OptixTraversableHandle handle;
	TransInfo* trans;
	uint2 size;
};
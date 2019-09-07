#pragma once
#include <OptiX/_Define.h>

namespace Define
{
	enum RayType
	{
		CloseRay = 0,
		AnyRay = 1,
	};
	struct RayData
	{
		float3 color;
		int depth;
		//float3 weight;
	};
}
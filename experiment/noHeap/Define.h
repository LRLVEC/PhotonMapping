#pragma once
#ifdef DefineDevice
#include <OptiX/_Define_7_Device.h>
struct TransInfo
{
	float4 row0;
	float4 row1;
	float4 row2;
	float3 r0;
	float z0;
};
#else
using TransInfo = CUDA::OptiX::Trans::TransInfo;
#endif
enum RayType
{
	RayRadiance = 0,
	ShadowRay = 1,
	RayCount
};
struct RayData
{
	float r, g, b;
};

struct LightSource
{
	float3 position;
	float3 power;
	float3 direction;
};

struct Photon
{
	float3 position;
	float3 dir;
	float3 energy;
	int primIdx;
};

struct PhotonHash
{
	Photon* pointer;
	int hashValue;

	bool operator < (const PhotonHash& a)
	{
		return hashValue < a.hashValue;
	}
};

struct PhotonPrd
{
	float3 energy;
	int startIdx;
	int numDeposits;	// number of times being recorded
	int depth;			// number of reflection
};

struct DebugData
{
	float3 position;
	int hashValue;
};

// data passed to Rt_RayGen
struct Rt_RayGenData
{
	
};

struct Rt_HitData
{
	float3* normals;
	float3* kds;
	LightSource* lightSource;
	Photon* photonMap; 
	int* NOLT;	// neighbour offset lookup table
	int* photonMapStartIdxs;
	DebugData* debugDatas;
};

#define PT_PHOTON_CNT (1 << 16)
#define PT_MAX_DEPTH 8
#define PT_MAX_DEPOSIT 8

struct Pt_RayGenData
{
	LightSource* lightSource;
	Photon* photons;
};

struct Pt_HitData
{
	float3* normals;
	float3* kds;
	Photon* photons;
};

struct Parameters
{
	float4* image;
	OptixTraversableHandle handle;
	TransInfo* trans;
	uint2 size;
	curandState* randState;
	float3 gridOrigin;
	int3 gridSize;
};


#define COLLECT_RAIDUS 0.2f

#define hash(position) ((int)floorf((position.z - paras.gridOrigin.z) / COLLECT_RAIDUS)) * paras.gridSize.x * paras.gridSize.y \
+ ((int)floorf((position.y - paras.gridOrigin.y) / COLLECT_RAIDUS)) * paras.gridSize.x \
+ ((int)floorf((position.x - paras.gridOrigin.x) / COLLECT_RAIDUS))
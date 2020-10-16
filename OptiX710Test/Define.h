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
	ConnectRay = 2,
	RayCount
};

enum eyeType
{
	LeftEye = 0,
	RightEye = 1,
	EyeCount
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

struct CameraRayData
{
	float3 position;
	float3 direction;
	int primIdx;
};

struct Photon
{
	float3 position;
	float3 energy;
	float2 dir;
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
	float3 v;
};

// data passed to Rt_RayGen
struct Rt_RayGenData
{
	CameraRayData* cameraRayDatas;
	DebugData* debugDatas;
};

struct Rt_HitData
{
	float3* normals;
	float3* kds;
	LightSource* lightSource;
	Photon* photonMap;
	int* NOLT;	// neighbour offset lookup table
	int* photonMapStartIdxs;
	CameraRayData* cameraRayDatas;
};

#define PT_PHOTON_CNT 640000
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
	float3* c_image;
	int* c_index;
	OptixTraversableHandle handle;
	TransInfo* trans;
	float3 invTrans[3];
	float3 rightEyePos;
	float z0;
	uint2 size;
	curandState* randState;
	float3 gridOrigin;
	int3 gridSize;
	eyeType eye;
};

#define COLLECT_RAIDUS 0.08f
#define HASH_GRID_SIDELENGTH COLLECT_RAIDUS

#define hash(position) ((int)floorf((position.z - paras.gridOrigin.z) / HASH_GRID_SIDELENGTH)) * paras.gridSize.x * paras.gridSize.y \
+ ((int)floorf((position.y - paras.gridOrigin.y) / HASH_GRID_SIDELENGTH)) * paras.gridSize.x \
+ ((int)floorf((position.x - paras.gridOrigin.x) / HASH_GRID_SIDELENGTH))

#define BLOCK_SIZE 8
#define BLOCK_SIZE2 64

#define CUDA_GATHER
//#define OPTIX_GATHER

//#define USE_SHARED_MEMORY

#define USE_CONNECTRAY
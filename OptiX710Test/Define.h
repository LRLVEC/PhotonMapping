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
	RayCount
};
struct RayData
{
	float r, g, b;
};

struct LightSource
{
	enum LightType { SPOT, SQUARE }type;
	float3 position;
	float3 power;
	// square lightsource
	float3 edge1;
	float3 edge2;
	float3 direction;
};

// information of the hit point of the camera ray
struct CameraRayHitData
{
	float3 position;	// position of the hit point
	float3 rayDir;		// direction of the camera ray
	int primIdx;		// index of the hitten primitive, -1 if miss
};

struct Photon
{
	float3 position;
	float3 normal;
	float3 dir;
	float3 energy;
	int axis;
};

struct PhotonPrd
{
	float3 energy;
	int startIdx;
	int numDeposits;	// number of times being recorded
	int depth;			// number of reflection
};

// data passed to Rt_RayGen
struct Rt_RayGenData
{
	CameraRayHitData* cameraRayHitDatas;	// to store the information of hit point
};

struct Rt_CloseHitData
{
	float3* normals;
	float3* kds;
	CameraRayHitData* cameraRayHitDatas;
};

struct Pt_RayGenData
{
	LightSource* lightSource;
	Photon* photons;
	float2* positionSeeds;
	float2* directionSeeds;
};

struct Pt_CloseHitData
{
	float3* normals;
	float3* kds;
	Photon* photons;
	float2* directionSeeds;	// random seeds for sampling
	float* RRseeds;
};

struct Parameters
{
	float4* image;
	OptixTraversableHandle handle;
	TransInfo* trans;
	uint2 size;
	int maxPhotonCnt;	// max number of records for a ray
	int maxDepth;		// max depth used in photon tracing
	uint2 pt_size;
};

struct Gt_RayGenData
{
	CameraRayHitData* cameraRayHitDatas;
	Photon* photonMap;
	float3* normals;
	float3* kds;
	LightSource* lightSource;
};



// macro used in Kd-Tree building
#define PPM_X ( 1 << 0 )
#define PPM_Y ( 1 << 1 )
#define PPM_Z ( 1 << 2 )
#define PPM_LEAF ( 1 << 3 )
#define PPM_NULL ( 1 << 4 )
#define PPM_INSHADOW ( 1 << 5 )
#define PPM_OVERFLOW ( 1 << 6 )
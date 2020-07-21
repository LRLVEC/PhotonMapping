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


// record the infomation of the hit point of the camera ray
struct CameraRayHitData
{
	float3 position;	// position the hit point
	float3 rayDir;		// direction of the camera ray
	int primIdx;		// index of the hitten primitive
	bool hit;
};


struct Rt_RayGenData
{
	RayData rayData;
	CameraRayHitData* cameraRayHitData;
};


struct Rt_CloseHitData
{
	float3* normals;
	float3* colors;
	float3* kd;
	CameraRayHitData* cameraRayHitData;
};


struct Parameters
{
	float4* image;
	OptixTraversableHandle handle;
	TransInfo* trans;
	uint2 size;
	int maxPhotonCnt;	// max number of record for a photon
	int maxDepth;		// max depth used in photon tracing
	uint2 pt_size;
};


struct PhotonRecord
{
	float3 position;
	float3 normal;
	float3 rayDir;
	float3 energy;
	int axis;
};


struct LightSource
{
	float3 position;
	float3 direction;
	float3 power;
	enum LightType { SPOT, AREA }type;
};


struct Pt_RayGenData
{
	LightSource* lightSource;
	PhotonRecord* photonRecord;
};


struct Pt_CloseHitData
{
	float3* normals;
	float3* colors;
	float3* kd;
	PhotonRecord* photonRecord;
};


struct PhotonPrd
{
	float3 energy;
	int pm_index;
	int numDeposits;	// number of times being recorded
	int depth;			// number of reflection
};


struct Gather_RayGenData
{
	CameraRayHitData* cameraRayHitData;
	PhotonRecord* photonMap;
	float3* normals;
	float3* colors;
	float3* kd;
	float3* attenKd;
	LightSource* lightSource;
};


struct Gather_CloseHitData
{
	float3* normals;	// not used
};


#define PPM_X ( 1 << 0 )
#define PPM_Y ( 1 << 1 )
#define PPM_Z ( 1 << 2 )
#define PPM_LEAF ( 1 << 3 )
#define PPM_NULL ( 1 << 4 )
#define PPM_INSHADOW ( 1 << 5 )
#define PPM_OVERFLOW ( 1 << 6 )
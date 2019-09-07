#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <OptiX/_Define.h>
#include "Define.h"
using namespace optix;
using namespace Define;
rtDeclareVariable(uint2, index, rtLaunchIndex, );
rtBuffer<float4, 2>result;
rtBuffer<float3>vertexBuffer;
rtBuffer<float3>vertexBufferIndexed;
rtBuffer<float3>normalBuffer;
rtBuffer<float3>colorBuffer;
rtBuffer<uint3>indexBuffer;
rtDeclareVariable(unsigned int, frame, , );
rtDeclareVariable(unsigned int, texid, , );
rtDeclareVariable(float3, color, , );
rtDeclareVariable(float, n, , );
rtDeclareVariable(float3, decay, , );
rtDeclareVariable(float, scatter, , );
rtDeclareVariable(float, cosTheta, , );
rtDeclareVariable(Trans, trans, , );
rtDeclareVariable(TransDepth, transDepth, , );
rtDeclareVariable(float, offset, , );
rtDeclareVariable(unsigned int, depthMax, , );
rtDeclareVariable(unsigned int, glassDepthMax, , );
rtDeclareVariable(unsigned int, russian, , );
rtDeclareVariable(rtObject, group, , );
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(RayData, rayData, rtPayload, );
rtDeclareVariable(float, l, rtIntersectionDistance, );
rtDeclareVariable(float3, normal, attribute normal, );
rtDeclareVariable(float2, texcoord, attribute texcoord, );
rtDeclareVariable(float3, colored, attribute colored, );

//rtTextureSampler<uchar4, 3, cudaReadModeNormalizedFloat> ahh;

RT_PROGRAM void rayAllocator()
{
	size_t2 size(result.size());
	int2 screen = make_int2(size.x, size.y);
	float2 ahh = random(index, screen, frame) + make_float2(index) - make_float2(size) / 2.0f;
	float4 d = make_float4(ahh, trans.z0, 0);
	Ray rayOrigin(trans.r0, normalize(trans.trans * d), CloseRay, offset);
	RayData rayDataOrigin;
	rayDataOrigin.depth = 0;
	//rayDataOrigin.weight = make_float3(1);
	rtTrace(group, rayOrigin, rayDataOrigin);
	if (frame)
		result[index] += make_float4(rayDataOrigin.color, 1.0f);
	else
		result[index] = make_float4(rayDataOrigin.color, 1.0f);
}
RT_PROGRAM void rayAllocatorDepth()
{
	size_t2 size(result.size());
	int2 screen = make_int2(size.x, size.y);
	float2 ahh = 2.0f * random(index, screen, frame) -
		make_float2(screen.x - int(index.x * 2), screen.y - int(index.y * 2));
	float2 delta(transDepth.D * randomCircle(random1(index, screen, frame)));
	float4 d = make_float4(ahh * transDepth.P - delta, transDepth.V, 0);
	Ray rayOrigin(transDepth.r0 + transDepth.trans * make_float4(delta, 0, 0),
		normalize(transDepth.trans * d), CloseRay, offset);
	RayData rayDataOrigin;
	rayDataOrigin.depth = 0;
	//rayDataOrigin.weight = make_float3(1);
	rtTrace(group, rayOrigin, rayDataOrigin);
	if (frame)
		result[index] += make_float4(rayDataOrigin.color, 1.0f);
	else
		result[index] = make_float4(rayDataOrigin.color, 1.0f);
}
RT_PROGRAM void metalAnyHit()
{
	rtTerminateRay();
}
RT_PROGRAM void metalCloseHit()
{
	float3 answer = make_float3(0);
	if (rayData.depth < depthMax)
	{
		Ray rayNow;
		rayNow.origin = ray.origin + l * ray.direction;
		RayData rayDataNow;
		rayDataNow.depth = rayData.depth + 1;
		rayNow.direction = ray.direction - 2 * dot(ray.direction, normal) * normal;
		rayNow.tmin = offset;
		rayNow.tmax = RT_DEFAULT_MAX;
		rtTrace(group, rayNow, rayDataNow);
		answer = rayDataNow.color * color;
	}
	rayData.color = answer * colored;
}
RT_PROGRAM void glassAnyHit()
{
	rtTerminateRay();
}
RT_PROGRAM void glassCloseHit()
{
	float3 answer = make_float3(0);
	if (rayData.depth < glassDepthMax)
	{
		float4 r = make_float4(1);
		float4 t = make_float4(1);
		float cosi1 = dot(ray.direction, normal);
		float nt(cosi1 < 0 ? n : 1 / n);
		float sini1 = sqrtf(1 - cosi1 * cosi1);
		float sini2 = sini1 / nt;
		Ray rayNow;
		rayNow.origin = ray.origin + l * ray.direction;
		RayData rayDataNow;
		rayDataNow.depth = rayData.depth + 1;
		if (sini2 < 1)
		{
			float cosi2 = sqrtf(1 - sini2 * sini2);
			if (sini2 <= 0.02)
			{
				float ahh = 4 * nt / ((nt + 1) * (nt + 1));
				t.w *= ahh;
				r.w *= 1 - ahh;
			}
			else
			{
				float a1 = nt * fabsf(cosi1) + cosi2;
				float a2 = fabsf(cosi1) + nt * cosi2;
				r.w *= (pow((nt * cosi2 - fabsf(cosi1)) / a2, 2) + pow((cosi2 - nt * fabsf(cosi1)) / a1, 2)) / 2;
				t.w *= 2 * cosi2 * (1 / pow(a1, 2) + 1 / pow(a2, 2)) * nt * fabsf(cosi1);
			}
			rayNow.direction = (ray.direction + (nt * ::copysignf(cosi2, cosi1) - cosi1) * normal) / nt;
			rayNow.tmin = offset;
			rayNow.tmax = RT_DEFAULT_MAX;
			rtTrace(group, rayNow, rayDataNow);
			if (cosi1 > 0) { float3 s(expf(-decay * l)); *(float3*)& t *= s; *(float3*)& r *= s; }
			answer += rayDataNow.color * make_float3(t) * t.w;
		}
		else
			*(float3*)& r = expf(-decay * l);
		rayNow.direction = ray.direction - 2 * cosi1 * normal;
		rtTrace(group, rayNow, rayDataNow);
		answer += rayDataNow.color * make_float3(r) * r.w;
	}
	rayData.color = answer * colored;
}
RT_PROGRAM void diffuseAnyHit()
{
	rtTerminateRay();
}
RT_PROGRAM void diffuseCloseHIt()
{
	float3 answer = make_float3(0);
	if (rayData.depth < depthMax)
	{
		float k(1);
		float2 seed(make_float2(ray.origin.y - sqrtf(frame), ray.direction.z + sqrtf(frame)));
		if (rayData.depth > russian)
		{
			if (random(seed) < 0.2f) { rayData.color = answer; return; }
			else k /= 0.8f;
		}
		float cosi1 = dot(ray.direction, normal);
		Ray rayNow;
		rayNow.origin = ray.origin + l * ray.direction;
		rayNow.tmin = offset;
		rayNow.tmax = RT_DEFAULT_MAX;
		RayData rayDataNow;
		rayDataNow.depth = rayData.depth + 1;
		rayNow.direction = randomDirectionCosN(cosi1 <= 0 ? normal : -normal, 1, seed);
		rtTrace(group, rayNow, rayDataNow);
		answer += rayDataNow.color * color * k;
	}
	rayData.color = answer * colored;
}
RT_PROGRAM void scatterAnyHit()
{
	rtTerminateRay();
}
RT_PROGRAM void scatterCloseHit()
{
	float3 answer = make_float3(0);
	float3 ratio;
	float ratioW(1);
	if (rayData.depth < depthMax)
	{
		float3 seed(make_float3(
			ray.origin.y - sqrtf(frame),
			ray.direction.z + sqrtf(frame),
			ray.direction.x + sqrtf(frame)
		));
		float gg(random(make_float2(seed)));
		if (rayData.depth > russian)
		{
			if (gg < 0.2f) { rayData.color = answer; return; }
			else ratioW /= 0.8f;
		}
		float cosi1 = dot(ray.direction, normal);
		Ray rayNow;
		rayNow.tmin = offset;
		rayNow.tmax = RT_DEFAULT_MAX;
		RayData rayDataNow;
		rayDataNow.depth = rayData.depth + 1;
		gg = random(make_float2(seed.y, seed.z));
		if (cosi1 > 0 && gg > expf(-l * scatter))
		{
			float4 ahh(randomScatter(ray.direction, l, scatter, seed));
			rayNow.origin = ray.origin + ahh.w * ray.direction;
			rayNow.direction = make_float3(ahh);
			ratio = scatterRatio * expf(-decay * ahh.w);
		}
		else
		{
			rayNow.origin = ray.origin + l * ray.direction;
			rayNow.direction = ray.direction;
			if (cosi1 > 0)ratio = expf(-decay * l);
			else ratio = { 1,1,1 };
		}
		rtTrace(group, rayNow, rayDataNow);
		answer += rayDataNow.color * ratio * ratioW;
	}
	rayData.color = answer * colored;
}
RT_PROGRAM void glassScatterAnyHit()
{
	rtTerminateRay();
}
RT_PROGRAM void glassScatterCloseHit()
{
	float3 answer = make_float3(0);
	if (rayData.depth < 20)
	{
		float3 seed(make_float3(
			ray.origin.y - sqrtf(frame),
			ray.direction.z + sqrtf(frame),
			ray.direction.x + sqrtf(frame)
		));
		float gg(random(make_float2(seed)));
		float ratioW(1);
		if (rayData.depth > russian)
		{
			if (gg < 0.2f) { rayData.color = answer; return; }
			else ratioW /= 0.8f;
		}
		float cosi1 = dot(ray.direction, normal);
		Ray rayNow;
		rayNow.tmin = offset;
		rayNow.tmax = RT_DEFAULT_MAX;
		RayData rayDataNow;
		rayDataNow.depth = rayData.depth + 1;
		gg = random(make_float2(seed.y, seed.z));
		if (cosi1 < 0 || gg < expf(-l * scatter))
		{
			float nt(cosi1 < 0 ? n : 1 / n);
			float sini1 = sqrtf(1 - cosi1 * cosi1);
			float sini2 = sini1 / nt;
			float4 r = make_float4(1);
			float4 t = make_float4(1);
			rayNow.origin = ray.origin + l * ray.direction;
			if (sini2 < 1)
			{
				float cosi2 = sqrtf(1 - sini2 * sini2);
				if (sini2 <= 0.02)
				{
					float ahh = 4 * nt / ((nt + 1) * (nt + 1));
					t.w *= ahh;
					r.w *= 1 - ahh;
				}
				else
				{
					float a1 = nt * fabsf(cosi1) + cosi2;
					float a2 = fabsf(cosi1) + nt * cosi2;
					r.w *= (pow((nt * cosi2 - fabsf(cosi1)) / a2, 2) + pow((cosi2 - nt * fabsf(cosi1)) / a1, 2)) / 2;
					t.w *= 2 * cosi2 * (1 / pow(a1, 2) + 1 / pow(a2, 2)) * nt * fabsf(cosi1);
				}
				rayNow.direction = (ray.direction + (nt * ::copysignf(cosi2, cosi1) - cosi1) * normal) / nt;
				rtTrace(group, rayNow, rayDataNow);
				if (cosi1 > 0) { float3 s(expf(-decay * l)); *(float3*)& t *= s; *(float3*)& r *= s; }
				answer += rayDataNow.color * make_float3(t) * t.w;
			}
			else
				*(float3*)& r = expf(-decay * l);
			rayNow.direction = ray.direction - 2 * cosi1 * normal;
			rtTrace(group, rayNow, rayDataNow);
			answer += rayDataNow.color * make_float3(r) * r.w;
			rayData.color = answer * colored;
			return;
		}
		else
		{
			float sini1 = sqrtf(1 - cosi1 * cosi1);
			float sini2 = sini1 / n;
			float cosi2 = sqrtf(1 - sini2 * sini2);
			float4 ahh(randomScatter((ray.direction + (n * ::copysignf(cosi2, cosi1) - cosi1) * normal) / n, l, scatter, seed));
			rayNow.origin = ray.origin + ahh.w * ray.direction;
			rayNow.direction = make_float3(ahh);
			rtTrace(group, rayNow, rayDataNow);
			rayData.color = rayDataNow.color * colored * expf(-decay * ahh.w) * ratioW;
			return;
		}
	}
}
RT_PROGRAM void lightAnyHit()
{
	rtTerminateRay();
}
RT_PROGRAM void lightCloseHit()
{
	if (dot(normal, ray.direction) < 0)
		rayData.color = color * colored;// *normalize(normal + make_float3(1));
	else
		rayData.color = { 0,0,0 };
}
RT_PROGRAM void torchAnyHit()
{
	rtTerminateRay();
}
RT_PROGRAM void torchCloseHit()
{
	float a(dot(ray.direction, normal));
	if (a < 0)
	{
		if (a < -cosTheta)rayData.color = color;
		else if (rayData.depth == 0)rayData.color = { 1, 1, 1 };
	}
	else rayData.color = { 0,0,0 };
}
RT_PROGRAM void miss()
{
	rayData.color = make_float3(rtTexCubemap<float4>(texid, ray.direction.x, ray.direction.y, ray.direction.z));
	//float3 source{ 0.9,0.4,-0.6 };
	//float3 dir(normalize(ray.origin - source));
	//if (dir.x < -0.999 && rayData.depth && dot(dir, ray.direction) < -0.9)
	//	rayData.color = { 10, 10, 10 };//
	//else
	//rayData.color = { 0 };
}
RT_PROGRAM void exception()
{
	result[index] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
}
RT_PROGRAM void attrib()
{
	unsigned int id = rtGetPrimitiveIndex();
	float3 p0 = vertexBuffer[3 * id];
	float3 p1 = vertexBuffer[3 * id + 1];
	float3 p2 = vertexBuffer[3 * id + 2];
	float3 d1 = p1 - p0;
	float3 d2 = p2 - p0;
	texcoord = rtGetTriangleBarycentrics();
	normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normalize(cross(d1, d2))));
	colored = { 1,1,1 };
}
RT_PROGRAM void attribIndexed()
{
	uint3 id = indexBuffer[rtGetPrimitiveIndex()];
	texcoord = rtGetTriangleBarycentrics();
	normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normalize(
		texcoord.x * normalBuffer[id.y] +
		texcoord.y * normalBuffer[id.z] +
		(1 - texcoord.x - texcoord.y) * normalBuffer[id.x])));
	colored = { 1,1,1 };
}
RT_PROGRAM void attribColored()
{
	unsigned int id = rtGetPrimitiveIndex();
	float3 p0 = vertexBuffer[3 * id];
	float3 p1 = vertexBuffer[3 * id + 1];
	float3 p2 = vertexBuffer[3 * id + 2];
	float3 d1 = p1 - p0;
	float3 d2 = p2 - p0;
	texcoord = rtGetTriangleBarycentrics();
	normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normalize(cross(d1, d2))));
	colored = colorBuffer[id];
}
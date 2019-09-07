#include <cstdio>
#include <cstdlib>
#define __OptiX__
#include <GL/_OpenGL.h>
#include <GL/_Window.h>
#include <GL/_Texture.h>
#include <OptiX/_OptiX.h>
#include <OptiX/_Define.h>
#include "Define.h"
#include <_Math.h>
#include <_Time.h>
#include <_Array.h>
#include <_STL.h>

namespace OpenGL
{
	namespace OptiX
	{
		using namespace Define;
		struct Scatter :RayTracer
		{
			struct Metal :Material
			{
				Variable<RTmaterial>color;
				Program closeHitProgram;
				Program anyHitProgram;
				CloseHit closeHit;
				AnyHit anyHit;
				Metal(RTcontext* _context, PTXManager& _pm)
					:
					Material(_context),
					color(&material, "color"),
					closeHitProgram(_context, _pm, "metalCloseHit"),
					anyHitProgram(_context, _pm, "metalAnyHit"),
					closeHit(&material, closeHitProgram, CloseRay),
					anyHit(&material, anyHitProgram, AnyRay)
				{
					closeHit.setProgram();
					anyHit.setProgram();
				}
			};
			struct Glass :Material
			{
				Variable<RTmaterial>color;
				Variable<RTmaterial>decay;
				Variable<RTmaterial>n;
				Program closeHitProgram;
				Program anyHitProgram;
				CloseHit closeHit;
				AnyHit anyHit;
				Glass(RTcontext* _context, PTXManager& _pm)
					:
					Material(_context),
					color(&material, "color"),
					decay(&material, "decay"),
					n(&material, "n"),
					closeHitProgram(_context, _pm, "glassCloseHit"),
					anyHitProgram(_context, _pm, "glassAnyHit"),
					closeHit(&material, closeHitProgram, CloseRay),
					anyHit(&material, anyHitProgram, AnyRay)
				{
					closeHit.setProgram();
					anyHit.setProgram();
				}
			};
			struct Diffuse :Material
			{
				Variable<RTmaterial>color;
				Program closeHitProgram;
				Program anyHitProgram;
				CloseHit closeHit;
				AnyHit anyHit;
				Diffuse(RTcontext* _context, PTXManager& _pm)
					:
					Material(_context),
					color(&material, "color"),
					closeHitProgram(_context, _pm, "diffuseCloseHIt"),
					anyHitProgram(_context, _pm, "diffuseAnyHit"),
					closeHit(&material, closeHitProgram, CloseRay),
					anyHit(&material, anyHitProgram, AnyRay)
				{
					closeHit.setProgram();
					anyHit.setProgram();
				}
			};
			struct Scattering :Material
			{
				Variable<RTmaterial>scatter;
				Variable<RTmaterial>decay;
				Program closeHitProgram;
				Program anyHitProgram;
				CloseHit closeHit;
				AnyHit anyHit;
				Scattering(RTcontext* _context, PTXManager& _pm)
					:
					Material(_context),
					scatter(&material, "scatter"),
					decay(&material, "decay"),
					closeHitProgram(_context, _pm, "scatterCloseHit"),
					anyHitProgram(_context, _pm, "scatterAnyHit"),
					closeHit(&material, closeHitProgram, CloseRay),
					anyHit(&material, anyHitProgram, AnyRay)
				{
					closeHit.setProgram();
					anyHit.setProgram();
				}
			};
			struct GlassScatter :Material
			{
				Variable<RTmaterial>decay;
				Variable<RTmaterial>n;
				Variable<RTmaterial>scatter;
				Program closeHitProgram;
				Program anyHitProgram;
				CloseHit closeHit;
				AnyHit anyHit;
				GlassScatter(RTcontext* _context, PTXManager& _pm)
					:
					Material(_context),
					decay(&material, "decay"),
					n(&material, "n"),
					scatter(&material, "scatter"),
					closeHitProgram(_context, _pm, "glassScatterCloseHit"),
					anyHitProgram(_context, _pm, "glassScatterAnyHit"),
					closeHit(&material, closeHitProgram, CloseRay),
					anyHit(&material, anyHitProgram, AnyRay)
				{
					closeHit.setProgram();
					anyHit.setProgram();
				}
			};
			struct Light :Material
			{
				Variable<RTmaterial>color;
				Program closeHitProgram;
				Program anyHitProgram;
				CloseHit closeHit;
				AnyHit anyHit;
				Light(RTcontext* _context, PTXManager& _pm)
					:
					Material(_context),
					color(&material, "color"),
					closeHitProgram(_context, _pm, "lightCloseHit"),
					anyHitProgram(_context, _pm, "lightAnyHit"),
					closeHit(&material, closeHitProgram, CloseRay),
					anyHit(&material, anyHitProgram, AnyRay)
				{
					closeHit.setProgram();
					anyHit.setProgram();
				}
			};
			struct Torch :Material
			{
				Variable<RTmaterial>color;
				Variable<RTmaterial>cosTheta;
				Program closeHitProgram;
				Program anyHitProgram;
				CloseHit closeHit;
				AnyHit anyHit;
				Torch(RTcontext* _context, PTXManager& _pm)
					:
					Material(_context),
					color(&material, "color"),
					cosTheta(&material, "cosTheta"),
					closeHitProgram(_context, _pm, "torchCloseHit"),
					anyHitProgram(_context, _pm, "torchAnyHit"),
					closeHit(&material, closeHitProgram, CloseRay),
					anyHit(&material, anyHitProgram, AnyRay)
				{
					closeHit.setProgram();
					anyHit.setProgram();
				}
			};
			DefautRenderer renderer;
			PTXManager pm;
			Context context;
			TransDepth trans;
			Program exception;
			Buffer resultBuffer;
			Buffer texBuffer;
			Metal metal;
			Glass glass;
			Diffuse diffuse;
			Scattering scatter;
			GlassScatter glassScatter;
			Light light;
			Torch torch;
			GeometryTriangles boxTriangles;
			GeometryTriangles bunnyTriangles;
			GeometryTriangles lightTriangles;
			GeometryTriangles torchTriangles;
			RTtexturesampler sampler;
			Variable<RTcontext> result;
			Variable<RTcontext> texTest;
			Variable<RTcontext> frame;
			Variable<RTcontext> texid;
			Variable<RTcontext> offset;
			Variable<RTcontext> depthMax;
			Variable<RTcontext> glassDepthMax;
			Variable<RTcontext> russian;
			//GeometryInstance instanceMetal;
			//GeometryInstance instanceGlass;
			GeometryInstance instanceBox;
			GeometryInstance instanceBunny;
			GeometryInstance instanceLight;
			GeometryInstance instanceTorch;
			//GeometryGroup geoGroupMetal;
			//GeometryGroup geoGroupGlass;
			GeometryGroup geoGroupBox;
			GeometryGroup geoGroupBunny;
			GeometryGroup geoGroupLight;
			GeometryGroup geoGroupTorch;
			//Transform transGlass;
			//Transform transDiffuse;
			//Transform transScatter;
			//Transform transLight;
			Group group;
			STL bunny;
			STL box;
			STL lightSource;
			STL torchSource;
			BMP testBMP;
			BMPCube skyBox;
			unsigned int frameNum;
			Scatter(::OpenGL::SourceManager* _sm, FrameScale const& _size)
				:
				renderer(_sm, _size),
				pm(&_sm->folder),
				context(pm, { {0,"rayAllocatorDepth"} }, { {CloseRay,"miss"} }, 2, 30),
				trans({ context, {30,0,5},{0.001,0.9,0.0005},{0.06},{0,0,0},700.0 }),
				exception(context, pm, "exception"),
				resultBuffer(context, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, renderer),
				texBuffer(context, RT_BUFFER_INPUT | RT_BUFFER_CUBEMAP, RT_FORMAT_UNSIGNED_BYTE4),
				metal(context, pm),
				glass(context, pm),
				diffuse(context, pm),
				scatter(context, pm),
				glassScatter(context, pm),
				light(context, pm),
				torch(context, pm),
				boxTriangles(context, pm, "attribColored", 1, RT_GEOMETRY_BUILD_FLAG_NONE,
					{
						{"vertexBuffer",RT_BUFFER_INPUT, RT_FORMAT_FLOAT3},
						{"colorBuffer",RT_BUFFER_INPUT, RT_FORMAT_FLOAT3}
					}),
				bunnyTriangles(context, pm, "attribIndexed", 1, RT_GEOMETRY_BUILD_FLAG_NONE,
					{
						{"vertexBufferIndexed",RT_BUFFER_INPUT, RT_FORMAT_FLOAT3},
						{"normalBuffer",RT_BUFFER_INPUT, RT_FORMAT_FLOAT3},
						{"indexBuffer",RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3}
					}),
				lightTriangles(context, pm, "attrib", 1, RT_GEOMETRY_BUILD_FLAG_NONE,
					{
						{"vertexBuffer",RT_BUFFER_INPUT, RT_FORMAT_FLOAT3}
					}),
				torchTriangles(context, pm, "attrib", 1, RT_GEOMETRY_BUILD_FLAG_NONE,
					{
						{"vertexBuffer",RT_BUFFER_INPUT, RT_FORMAT_FLOAT3}
					}),
				result(context, "result"),
				texTest(context, "ahh"),
				frame(context, "frame"),
				texid(context, "texid"),
				offset(context, "offset"),
				depthMax(context, "depthMax"),
				glassDepthMax(context, "glassDepthMax"),
				russian(context, "russian"),
				//instanceMetal(context),
				//instanceGlass(context),
				instanceBox(context),
				instanceBunny(context),
				instanceLight(context),
				instanceTorch(context),
				//geoGroupMetal(context, Acceleration::Trbvh),
				//geoGroupGlass(context, Acceleration::Trbvh),
				geoGroupBox(context, Acceleration::Sbvh),
				geoGroupBunny(context, Acceleration::Sbvh),
				geoGroupLight(context, Acceleration::Sbvh),
				geoGroupTorch(context, Acceleration::Sbvh),
				//transGlass(context),
				//transDiffuse(context),
				//transScatter(context),
				//transLight(context),
				group(context, "group", Acceleration::Sbvh),
				bunny(pm.folder->find("resources/scene/bunny.stl").readSTL()),
				box(pm.folder->find("resources/scene/box.stl").readSTL()),
				lightSource(pm.folder->find("resources/scene/light.stl").readSTL()),
				torchSource(pm.folder->find("resources/scene/torch.stl").readSTL()),
				testBMP("resources/lightSource.bmp"),
				skyBox("resources/room/"),
				frameNum(0)
			{
				renderer.prepare();
				//context.printDeviceInfo();
				context.pringStackSize();
				trans.init(_size);
				resultBuffer.setSize(_size.w, _size.h);
				rtContextSetExceptionProgram(context, 0, exception);
				//RTtexturesampler sampler;
				//rtContextSetPrintEnabled(context, 1);
				//rtContextSetPrintBufferSize(context, 4096);
				texBuffer.readCube(skyBox);
				texBuffer.readBMP(testBMP);
				rtTextureSamplerCreate(context, &sampler);
				rtTextureSamplerSetWrapMode(sampler, 0, RT_WRAP_CLAMP_TO_EDGE);
				rtTextureSamplerSetWrapMode(sampler, 1, RT_WRAP_CLAMP_TO_EDGE);
				rtTextureSamplerSetFilteringModes(sampler, RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
				rtTextureSamplerSetIndexingMode(sampler, RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
				rtTextureSamplerSetReadMode(sampler, RT_TEXTURE_READ_NORMALIZED_FLOAT);
				rtTextureSamplerSetMaxAnisotropy(sampler, 1.0f);
				rtTextureSamplerSetBuffer(sampler, 0, 0, texBuffer);
				int tex_id;
				rtTextureSamplerGetId(sampler, &tex_id);
				texid.set1u(tex_id);

				boxTriangles.addSTL("vertexBuffer", box, box.triangles.length);
				bunnyTriangles.addSTL("vertexBufferIndexed", "normalBuffer", "indexBuffer", bunny);
				lightTriangles.addSTL("vertexBuffer", lightSource, lightSource.triangles.length);
				torchTriangles.addSTL("vertexBuffer", torchSource, torchSource.triangles.length);
				Buffer& boxColorBuffer(boxTriangles.buffers["colorBuffer"].buffer);
				boxColorBuffer.setSize(box.triangles.length);
				Math::vec3<float>* x((Math::vec3<float>*)boxColorBuffer.map());
				x[0] = x[1] = { 1,0,0 };
				for (int c0(2); c0 < box.triangles.length; ++c0)
					x[c0] = { 1,1,1 };
				x[4] = x[5] = { 0,1,0 };
				boxColorBuffer.unmap();
				//instanceMetal.setTriangles(trianglesIndexed);
				//instanceGlass.setTriangles(trianglesIndexed);
				instanceBox.setTriangles(boxTriangles);
				instanceBunny.setTriangles(bunnyTriangles);
				instanceLight.setTriangles(lightTriangles);
				instanceTorch.setTriangles(torchTriangles);
				//instanceMetal.setMaterial({ &glass });
				//instanceGlass.setMaterial({ &glass });
				instanceBox.setMaterial({ &diffuse });
				instanceBunny.setMaterial({ &glass });
				instanceLight.setMaterial({ &light });
				instanceTorch.setMaterial({ &torch });
				//geoGroupMetal.setInstance({ &instanceMetal });
				//geoGroupGlass.setInstance({ &instanceGlass });
				geoGroupBox.setInstance({ /*&instanceBox,*/&instanceBunny,&instanceLight,&instanceTorch });
				//geoGroupBunny.setInstance({ &instanceBunny });
				//geoGroupLight.setInstance({ &instanceLight });

				/*transGlass.setMat({
					{1, 0, 0, 10},
					{0, 1, 0, 0},
					{0, 0, 1, 0},
					{0, 0, 0, 1}
					});
				transDiffuse.setMat({
					{1, 0, 0, 0},
					{0, 1, 0, 10},
					{0, 0, 1, 0},
					{0, 0, 0, 1}
					});
				transScatter.setMat({
					{1, 0, 0, 10},
					{0, 1, 0, 10},
					{0, 0, 1, 0},
					{0, 0, 0, 1}
					});
				transLight.setMat({
					{1, 0, 0, 2.10},
					{0, 1, 0, 2.10},
					{0, 0, 1, 10},
					{0, 0, 0, 1}
					});
				*/
				//transGlass.setChild(geoGroupGlass);
				//transDiffuse.setChild(geoGroupDiffuse);
				//transScatter.setChild(geoGroupScatter);
				//transLight.setChild(geoGroupLight);

				group.setGeoGroup({ geoGroupBox });
				result.setObject(resultBuffer);
				metal.color.set3f(1.0f, 1.0f, 1.0f);
				glass.color.set3f(1.0f, 1.0f, 1.0f);
				glass.decay.set3f(0.2, 0.2, 0.2);
				glass.n.set1f(1.5);
				diffuse.color.set3f(0.9, 0.9, 0.9);
				scatter.scatter.set1f(3);
				scatter.decay.set3f(0, 0, 0);
				glassScatter.n.set1f(1.5);
				glassScatter.decay.set3f(0.3, 0, 0.3);
				glassScatter.scatter.set1f(5);
				light.color.set3f(20, 20, 20);
				torch.color.set3f(5000, 5000, 5000);
				torch.cosTheta.set1f(cosf(5.f * Math::Pi / 180.f));
				offset.set1f(1e-5f);
				depthMax.set1u(context.maxDepth - 1);
				glassDepthMax.set1u(8);
				russian.set1u(5);
				context.validate();
			}
			virtual void run()override
			{
				trans.operate();
				if (trans.updated)
				{
					frameNum = 0;
					trans.updated = false;
				}
				else ++frameNum;
				frame.set1u(frameNum);
				FrameScale size(renderer.size());
				context.launch(0, size.w, size.h);
				++frameNum;
				frame.set1u(frameNum);
				context.launch(0, size.w, size.h);
				renderer.updated = true;
				renderer.use();
				renderer.run();
			}
			virtual void resize(FrameScale const& _size)override
			{
				trans.resize(_size);
				resultBuffer.unreg();
				renderer.resize(_size);
				resultBuffer.setSize(_size.w, _size.h);
				resultBuffer.reg();
			}
			virtual void terminate()override
			{
				boxTriangles.destory();
				resultBuffer.destory();
				//instanceGlass.destory();
				//geoGroupGlass.destory();
				//geoGroupMetal.destory();
				group.destory();
				context.destory();
			}
		};
	}
	struct RayTracing :OpenGL
	{
		SourceManager sm;
		OptiX::Scatter monteCarlo;
		RayTracing(FrameScale const& _size)
			:
			sm(),
			monteCarlo(&sm, _size)
		{
		}
		virtual void init(FrameScale const& _size) override
		{
			monteCarlo.resize(_size);
		}
		virtual void run() override
		{
			monteCarlo.run();
		}
		void terminate()
		{
			monteCarlo.terminate();
		}
		virtual void frameSize(int _w, int _h)override
		{
			monteCarlo.resize({ _w,_h });
		}
		virtual void framePos(int, int) override {}
		virtual void frameFocus(int) override {}
		virtual void mouseButton(int _button, int _action, int _mods)override
		{
			switch (_button)
			{
				case GLFW_MOUSE_BUTTON_LEFT:monteCarlo.trans.mouse.refreshButton(0, _action); break;
				case GLFW_MOUSE_BUTTON_MIDDLE:monteCarlo.trans.mouse.refreshButton(1, _action); break;
				case GLFW_MOUSE_BUTTON_RIGHT:monteCarlo.trans.mouse.refreshButton(2, _action); break;
			}
		}
		virtual void mousePos(double _x, double _y)override
		{
			monteCarlo.trans.mouse.refreshPos(_x, _y);
		}
		virtual void mouseScroll(double _x, double _y)override
		{
			if (_y != 0.0)
				monteCarlo.trans.scroll.refresh(_y);
		}
		virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods) override
		{
			{
				switch (_key)
				{
					case GLFW_KEY_ESCAPE:
						if (_action == GLFW_PRESS)
							glfwSetWindowShouldClose(_window, true);
						break;
					case GLFW_KEY_A:monteCarlo.trans.key.refresh(0, _action); break;
					case GLFW_KEY_D:monteCarlo.trans.key.refresh(1, _action); break;
					case GLFW_KEY_W:monteCarlo.trans.key.refresh(2, _action); break;
					case GLFW_KEY_S:monteCarlo.trans.key.refresh(3, _action); break;
					case GLFW_KEY_UP:monteCarlo.trans.persp.increaseV(0.02); break;
					case GLFW_KEY_DOWN:monteCarlo.trans.persp.increaseV(-0.02); break;
					case GLFW_KEY_RIGHT:monteCarlo.trans.persp.increaseD(0.01); break;
					case GLFW_KEY_LEFT:monteCarlo.trans.persp.increaseD(-0.01); break;
				}
			}
		}
	};
}


int main()
{
	OpenGL::OpenGLInit init(4, 5);
	Window::Window::Data winParameters
	{
		"Scatter",
		{
			{1080,1080},
			true,false
		}
	};
	Window::WindowManager wm(winParameters);
	OpenGL::RayTracing scatter({ 1080,1080 });
	wm.init(0, &scatter);
	//init.printRenderer();
	glfwSwapInterval(0);
	FPS fps;
	fps.refresh();
	while (!wm.close())
	{
		wm.pullEvents();
		wm.render();
		wm.swapBuffers();
		//fps.refresh();
		//fps.printFPS(1);
	}
	scatter.terminate();
	return 0;
}


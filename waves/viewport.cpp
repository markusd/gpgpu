/**
 * @author Markus Holtermann
 * @date May 14, 2011
 * @file gui/renderwidget.cpp
 */

#ifdef __WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <viewport.hpp>
#include <qutils.hpp>

#include <m3d/m3d.hpp>

#include <opengl/shader.hpp>
#include <opengl/texture.hpp>

#include <util/inputadapters.hpp>
#include <util/config.hpp>
#include <util/tostring.hpp>

#include <QtCore/QString>
#include <QtGui/QWheelEvent>
#include <QtGui/QWidget>
#include <QtOpenGL/QGLWidget>

#include <CL/cl.hpp>
#include <opencl/oclutil.hpp>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <waves.cuh>
#endif

#include <iostream>


using namespace m3d;



// OpenCL vars
/*
cl_platform_id clPlatform;
cl_device_id clDevice;
cl_context clContext;
cl_command_queue clQueue;
cl_program clProgram;
cl_kernel clKernel;
*/

cl::Platform clPlatform;
std::vector<cl::Device> clDevices;
cl::Context clContext;
cl::CommandQueue clQueue;
cl::Program clProgram;
cl::Kernel clKernel;
cl::Buffer clOut;
cl::Image2DGL clTexture;
cl::Buffer clPos;
cl::BufferRenderGL clRenderBuffer;

GLuint glFBO;
GLuint glRB;

#ifdef USE_CUDA

#define MIN(a,b) ((a < b) ? a : b)
#define MAX(a,b) ((a > b) ? a : b)

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8 },
	  { 0x11,  8 },
	  { 0x12,  8 },
	  { 0x13,  8 },
	  { 0x20, 32 },
	  { 0x21, 48 },
	  {   -1, -1 } 
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
}

inline int cutGetMaxGflopsDeviceId()
{
	int current_device   = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device  = 0;
	int device_count     = 0, best_SM_arch     = 0;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount( &device_count );
	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major > 0 && deviceProp.major < 9999) {
			best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
		    sm_per_multiproc = 1;
		} else {
			sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		if( compute_perf  > max_compute_perf ) {
            // If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 ) {
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch) {	
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			} else {
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}

#endif

Viewport::Viewport(QWidget* parent) :
	QGLWidget(parent)
{
	setFocusPolicy(Qt::WheelFocus);

	m_mouseAdapter.addListener(this);
	m_xoffs = m_yoffs = 0.0f;
	m_mode = CPU;

	setMinimumWidth(WIDTH);
	setMinimumHeight(HEIGHT);
	setMaximumWidth(WIDTH);
	setMaximumHeight(HEIGHT);
#ifndef CPU_FLOAT
	m_textureData = (unsigned char *)malloc(WIDTH * HEIGHT * 3 * sizeof(unsigned char));
#else
	m_textureData = (float *)malloc(WIDTH * HEIGHT * 3 * sizeof(float));
#endif
//#ifndef GL_INTEROP
	m_clTextureData = (float *)malloc(WIDTH * HEIGHT * 4 * sizeof(float));
//#endif





	m_timer = new QTimer(this);
	connect(m_timer, SIGNAL(timeout()), this, SLOT(updateGL()));
}

void Viewport::initializeGL()
{
	GLenum gerr = glewInit();
	if (gerr != GLEW_OK) {
		util::ErrorAdapter::instance().displayErrorMessage("Could not initialize GLEW!");
		exit(1);
	}

	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	// wave description texture
	m_waveData = ogl::__Texture::create();
	m_waveData->setFilter(GL_NEAREST, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 32, 32, 0, GL_RGBA, GL_FLOAT, NULL);

	// CPU output texture
	m_texture = ogl::__Texture::create();
	m_texture->setFilter(GL_NEAREST, GL_NEAREST);
#ifdef GL_INTEROP
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, m_clTextureData);
#else
#ifdef CPU_FLOAT
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, m_textureData);
#else
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, m_textureData);
#endif
#endif

#ifdef GL_INTEROP
#ifdef GL_FBO
	glGenFramebuffersEXT(1, &glFBO);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, glFBO);

	glGenRenderbuffersEXT(1, &glRB);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, glRB);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGBA, WIDTH, HEIGHT);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, glRB);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
#endif
#endif

	// displaylist
	m_displayList = glGenLists(1);
	glNewList(m_displayList, GL_COMPILE);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glBegin(GL_QUADS);
			glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
			glTexCoord2f(1.0f, 0.0f); glVertex2f(WIDTH, 0.0f);
			glTexCoord2f(1.0f, 1.0f); glVertex2f(WIDTH, HEIGHT);
			glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, HEIGHT);
		glEnd();
	glEndList();


	/**
	 * Setup OpenCL
	 */

	try {
#ifdef GL_INTEROP
#ifdef _WIN32
		cl_context_properties glContext = (cl_context_properties)wglGetCurrentContext();
		cl_context_properties hDC = (cl_context_properties)wglGetCurrentDC();
#endif
#else
		cl_context_properties glContext = NULL;
		cl_context_properties hDC = NULL;
#endif

		ocl::createContext(CL_DEVICE_TYPE_ALL, glContext, hDC, clPlatform, clDevices, clContext, clQueue);
		std::string info("");
		cl_int clError;

#ifdef GL_INTEROP
#ifdef GL_FBO
		clRenderBuffer = cl::BufferRenderGL(clContext, CL_MEM_WRITE_ONLY, glRB, &clError);
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;
#else
		clTexture = cl::Image2DGL(clContext, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, m_texture->m_textureID, &clError);
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;
#endif
#else
		clOut = cl::Buffer(clContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, WIDTH * HEIGHT * 4 * sizeof(float), m_clTextureData, &clError);
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;
#endif

		clPos = cl::Buffer(clContext, CL_MEM_READ_ONLY, 32 * 2 * sizeof(float), NULL, &clError);
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;


	} catch (const cl::Error& error) {
		std::cout << "OpenCL Error 1: " << error.what() << " (" << error.err() << ")" << std::endl;
	}

	/**
	 * Setup CUDA
	 */

#ifdef USE_CUDA

#ifndef CUDA_GL_INTEROP
	cudaError_t err = cudaSetDevice(cutGetMaxGflopsDeviceId());
	std::cout << cudaGetErrorString( err ) << std::endl;
#else
	cudaError_t err = cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
	std::cout << cudaGetErrorString( err ) << std::endl;
#endif

	//m_cuHostTextureData = (float *)malloc(WIDTH * HEIGHT * 4 * sizeof(float));
	cudaMallocHost((void**)&m_cuHostTextureData, WIDTH * HEIGHT * 4 * sizeof(float));

	m_cuTexture = ogl::__Texture::create();
	m_cuTexture->setFilter(GL_NEAREST, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, m_cuHostTextureData);

#ifdef CUDA_GL_INTEROP
	err = cudaGraphicsGLRegisterImage(&m_cuTextureResource, m_cuTexture->m_textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	std::cout << cudaGetErrorString( err ) << std::endl;
#endif

	cudaMalloc((void**)&m_cuTextureData, WIDTH * HEIGHT * 4 * sizeof(float));
	cudaMalloc((void**)&m_cuPosData, 32 * 2 * sizeof(float));
#endif

	// create shader and kernel
	createKernel();
	createShader();

	ogl::__Shader::unbind();

	m_clock.reset();
}

void Viewport::resizeGL(int width, int height)
{
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0.0f, 0.0f, width, height);
	glOrtho(0.0f, width, 0.0f, height, 0.0f, 32.0f);
	glMatrixMode(GL_MODELVIEW);

}

void Viewport::paintGL()
{
	static int frames = 0;
	static float fps_time = 0.0f;

	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	float dt = m_clock.get();

	switch (m_mode) {
		case SHADER:
			if (m_waves.size() > 0) {
				m_shader->setUniform1f("timer", dt);
				m_shader->setUniform1f("xoffs", m_xoffs);
				m_shader->setUniform1f("yoffs", m_yoffs);
			}
			glCallList(m_displayList);
			break;
		case CPU:
			createTextureCPU(dt);
			glCallList(m_displayList);
			break;
		case OPENCL:
			createTextureOpenCL(dt);
#ifdef GL_FBO
			glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, glFBO);
			glBlitFramebufferEXT(0, 0, WIDTH, HEIGHT, 0, 0, WIDTH, HEIGHT, GL_COLOR_BUFFER_BIT, GL_NEAREST);
#else
			glCallList(m_displayList);
#endif
			break;
#ifdef USE_CUDA
		case CUDA:
			createTextureCuda(dt);
			glCallList(m_displayList);
			break;
#endif
	}


	frames++;
	if (dt - fps_time >= 1.0f) {
		//std::cout << dt << ", " << fps_time << ", " << frames << std::endl;
		fps_time = dt;
		emit framesPerSecondChanged(frames);
		frames = 0;
	}
}

void Viewport::createTextureOpenCL(float dt)
{
	if (m_waves.size() == 0)
		return;
	
	float wx = m_waves.front().pos.x;
	float wy = m_waves.front().pos.y;

	size_t globalWorkSize = WIDTH * HEIGHT;
	size_t localWorkSize = 256;
	cl_int n = WIDTH * HEIGHT;

	try {
		cl_int error = CL_SUCCESS;

#ifdef GL_INTEROP
#ifdef GL_FBO
		clKernel.setArgs(clRenderBuffer(), WIDTH, dt);
		std::vector<cl::Memory> objects;
		objects.push_back(clRenderBuffer);
#else
		clKernel.setArgs(clTexture(), WIDTH, dt);
		std::vector<cl::Memory> objects;
		objects.push_back(clTexture);
#endif
		clQueue.enqueueAcquireGLObjects(&objects, NULL, NULL);
#else
		clKernel.setArgs(clOut(), WIDTH, dt);
#endif

		clQueue.enqueueNDRangeKernel(clKernel, cl::NullRange,
			cl::NDRange(ocl::roundGlobalSize(16, WIDTH), ocl::roundGlobalSize(16, HEIGHT)), cl::NDRange(16,16), NULL, NULL);

		
#ifdef GL_INTEROP
		clQueue.enqueueReleaseGLObjects(&objects, NULL, NULL);
#else
		clQueue.enqueueReadBuffer(clOut, false, 0, WIDTH*HEIGHT*4*sizeof(float), m_clTextureData, NULL, NULL);
#endif

		clQueue.finish();

		glEnable(GL_TEXTURE_2D);
		m_texture->bind();

#ifndef GL_INTEROP
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT, m_clTextureData);
#endif

		int err = glGetError();
		if (err != GL_NO_ERROR)
			std::cout << gluErrorString(err) << std::endl;

	} catch (const cl::Error& error) {
		std::cout << "OpenCL Error 2: " << error.what() << " (" << error.err() << ")" << std::endl;
	}
}

void Viewport::createTextureCPU(float dt)
{
	typedef Vec3<unsigned char> Vec3ub;
	glEnable(GL_TEXTURE_2D);
	m_texture->bind();

	float dist = 0.0f;
	float amp = 0.0f;
	Vec2f TexCoord;

	dt *= 0.33333333f;
#ifndef CPU_FLOAT
	float waves = 637.5f / (float)m_waves.size();
#else
	float waves = 2.5f / (float)m_waves.size();
#endif
	if (m_waves.size() > 0) 
	for (int x = 0; x < WIDTH; ++x) {
		for (int y = 0; y < HEIGHT; ++y) {
#ifndef CPU_FLOAT
			Vec3ub* c = (Vec3ub *)&m_textureData[(x + y * WIDTH) * 3];
#else
			Vec3f* c = (Vec3f *)&m_textureData[(x + y * WIDTH) * 3];
#endif

			amp = 0.0f;

			TexCoord.x = (float)x / (float)WIDTH + m_xoffs;
			TexCoord.y = (float)y / (float)HEIGHT + m_yoffs;


			for (int i = 0; i < m_waves.size(); ++i) {
				dist = (TexCoord - m_waves[i].pos).len();
				amp += waves * sinf(6.2831f * (dt - dist * 5.0f));
			}
#ifndef CPU_FLOAT
			int iamp = abs((int)amp) * 2;

			c->x = c->y = c->z = 0;

			if (iamp <= 255) {
				c->x = 255;
				c->y = iamp;
			} else if (iamp <= 510) {
				c->x = 500 - iamp;
				c->y = 255;
			} else if (iamp <= 765) {
				c->y = 255;
				c->z = iamp - 510;
			} else if (iamp <= 1020) {
				c->y = 1020 - iamp;
				c->z = 255;
			} else {
				c->x = iamp - 1024;
				c->z = 255;
			}

			c->x = 255 - c->x;
			c->y = 255 - c->y;
			c->z = 255 - c->z;
#else
				float r = 0.0f;
				float g = 0.0f;
				float b = 0.0f;
				amp = 2.0f * (amp < 0.0f ? -amp : amp);
				if (amp <= 1.0f) {
					r = 1.0f;
					g = amp;
				} else if (amp <= 2.0f) {
					r = (1.96f - amp);
					g = 1.0f;
				} else if (amp <= 3.0f) {
					g = 1.0f;
					b = (amp - 2.0f);
				} else if (amp <= 4.0f) {
					g = (4.0f - amp);
					b = 1.0f;
				} else {
					r = (amp - 4.01f);
					b = 1.0f;
				}

				c->x = 1.0f - r;
				c->y = 1.0f - g;
				c->z = 1.0f - b;
#endif
			//std::cout << (int)c->x << std::endl;
			//std::cout << (int)c->y << std::endl;
			//std::cout << (int)c->z << std::endl << std::endl;
		}
	}
#ifndef CPU_FLOAT
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, m_textureData);
#else
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_FLOAT, m_textureData);
#endif
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, m_textureData);
}

#ifdef USE_CUDA
void Viewport::createTextureCuda(float dt)
{
#ifdef CUDA_GL_INTEROP
	cuda_launch_waves(m_cuTextureData, WIDTH, dt, m_waves.size(), m_cuPosData);

	cudaError_t err = cudaGraphicsMapResources(1, &m_cuTextureResource, 0);
	cudaArray *in_array;

	err = cudaGraphicsSubResourceGetMappedArray(&in_array, m_cuTextureResource, 0, 0);
	//std::cout << cudaGetErrorString(err) << std::endl;
	err = cudaMemcpyToArray(in_array, 0, 0, m_cuTextureData, WIDTH * HEIGHT * sizeof(float) * 4, cudaMemcpyDeviceToDevice);
	err = cudaGraphicsUnmapResources(1, &m_cuTextureResource, 0);

	glEnable(GL_TEXTURE_2D);
	m_cuTexture->bind();
#else
	cuda_launch_waves(m_cuTextureData, WIDTH, dt, m_waves.size(), m_cuPosData);
	cudaMemcpy(m_cuHostTextureData, m_cuTextureData, WIDTH * HEIGHT * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	glEnable(GL_TEXTURE_2D);
	m_cuTexture->bind();
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT, m_cuHostTextureData);
#endif
}
#endif

void Viewport::setComputationMode(ComputationMode mode)
{
	if (m_mode == mode)
		return;

	m_mode = mode;

	if (m_mode == SHADER) {
		m_shader->bind();
		glDisable(GL_TEXTURE_2D);
		m_waveData->bind();
	} else if (m_mode == CPU) {
		glEnable(GL_TEXTURE_2D);
		m_texture->bind();
		ogl::__Shader::unbind();
#ifndef CPU_FLOAT
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, NULL);
#else
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_FLOAT, NULL);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
#endif
	} else if  (m_mode == OPENCL) {
		glEnable(GL_TEXTURE_2D);
		m_texture->bind();
		ogl::__Shader::unbind();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
		int err = glGetError();
		if (err != GL_NO_ERROR)
			std::cout << "setComputationMode: " << gluErrorString(err) << std::endl;
	}
#ifdef USE_CUDA
	else if  (m_mode == CUDA) {
		glEnable(GL_TEXTURE_2D);
		m_cuTexture->bind();
		ogl::__Shader::unbind();
	}
#endif
}

void Viewport::mouseMove(int x, int y)
{
	if (m_mouseAdapter.isDown(util::RIGHT)) {
		m_xoffs -= (x - m_mouseAdapter.getX()) / (float)width();
		m_yoffs += (y - m_mouseAdapter.getY()) / (float)height();
	}
}

void Viewport::mouseButton(util::Button button, bool down, int x, int y)
{
	if (button == util::LEFT && !down) {
		float mx = (float)x / (float)width() + m_xoffs;
		float my = ((float)height() - (float)y) / (float)height() + m_yoffs;

		Wave wave(mx, my);
		m_waves.push_back(wave);

		std::cout << mx << ", " << my << std::endl;

		float data[4] = { mx, my, 0.0f, 1.0f };
		m_waveData->bind();
		glTexSubImage2D(GL_TEXTURE_2D, 0, m_waves.size() - 1, 0, 1, 1, GL_RGBA, GL_FLOAT, data);

		createKernel();
		createShader();

		try {
			clQueue.enqueueWriteBuffer(clPos, CL_TRUE, (m_waves.size() - 1) * 2 * sizeof(float), 2 * sizeof(float), data, NULL, NULL);
			clKernel.setArg(3, clPos());
		} catch (const cl::Error& error) {
			std::cout << "OpenCL Error 3: " << error.what() << " (" << error.err() << ")" << std::endl;
		}

#ifdef USE_CUDA
		cudaMemcpy(m_cuPosData, &m_waves[0], m_waves.size() * 2 * sizeof(float), cudaMemcpyHostToDevice);
#endif

		if (m_mode != SHADER)
			ogl::__Shader::unbind();
	}
}

void Viewport::createKernel()
{
	cl_int clError = CL_SUCCESS;

	std::string code = "";

	if (m_waves.size() > 0) {
		code = "__constant const float waves = 2.5f / ";
		code += util::toString(m_waves.size());
		code += ".0f;\n\n";
		code +=
#ifdef GL_INTEROP
			"__kernel void wave(__write_only image2d_t out, int width, float dt, __constant float* pos) {\n"
#else
			"__kernel void wave(__global float* out, int width, float dt, __constant float* pos) {\n"
#endif
				"int x = get_global_id(0);\n"
				"int y = get_global_id(1);\n"
				"dt *= 0.33333333f;\n"
				"float dx = 0.0f;\n"
				"float dy = 0.0f;\n"
				"float amp = 0.0f;\n"
				"float dist = 0.0f;\n";
		
		for (int i = 0; i < m_waves.size(); ++i) {
			code += "dx = (float)x / (float)width - pos[";
			code += util::toString(i);
			code += "*2];\n"
					"dy = (float)y / (float)width - pos[";
			code += util::toString(i);
			code += "*2+1];\n"
					"dist = native_sqrt(dx * dx + dy * dy);\n"
					"amp += waves * native_sin(6.2831f * (dt - dist * 5.0f));\n";
		}


		code +=
				"float r = 0.0f;\n"
				"float g = 0.0f;\n"
				"float b = 0.0f;\n"
				"amp = 2.0f * (amp < 0.0f ? -amp : amp);\n"
				"if (amp <= 1.0f) {\n"
					"r = 1.0f;\n"
					"g = amp;\n"
				"} else if (amp <= 2.0f) {\n"
					"r = (1.96f - amp);\n"
					"g = 1.0f;\n"
				"} else if (amp <= 3.0f) {\n"
					"g = 1.0f;\n"
					"b = (amp - 2.0f);\n"
				"} else if (amp <= 4.0f) {\n"
					"g = (4.0f - amp);\n"
					"b = 1.0f;\n"
				"} else {\n"
					"r = (amp - 4.01f);\n"
					"b = 1.0f;\n"
				"}\n"
#ifdef GL_INTEROP
				"write_imagef(out, (int2)(x, y), (float4)(1.0f - r, 1.0f - g, 1.0f - b, 0.0f));\n"
#else
				"out[(x + y * width) * 4] = 1.0f - r;\n"
				"out[(x + y * width) * 4 + 1] = 1.0f - g;\n"
				"out[(x + y * width) * 4 + 2] = 1.0f - b;\n"
				"out[(x + y * width) * 4 + 3] = 0.0f;\n"
#endif
			"}\n";
	} else {
		code =
			"__kernel void wave(__global float* out, int width, float dt, __constant float* pos) {\n"
			"}\n";
	}
	std::cout << code << std::endl;

	try {
		cl::Program::Sources source(1, std::make_pair(code.c_str(), code.size()));
		clProgram = cl::Program(clContext, source);
		clProgram.build(clDevices, "-cl-fast-relaxed-math");

		std::string info("");
		clProgram.getBuildInfo(clDevices[0], CL_PROGRAM_BUILD_LOG, &info);
		if (info.size() > 0)
			std::cout << "Build log: " << info << std::endl;

		clKernel = cl::Kernel(clProgram, "wave", &clError);

#ifdef GL_INTEROP
//		clKernel.setArgs(clTexture(), WIDTH, 0.0f, clPos());
#else
//		clKernel.setArgs(clOut(), WIDTH, 0.0f, clPos());
#endif

	} catch (const cl::Error& err) {
		std::cout << "OpenCL Error 4: " << err.what() << " (" << err.err() << ")" << std::endl;
		std::string info("");
		clProgram.getBuildInfo(clDevices[0], CL_PROGRAM_BUILD_LOG, &info);
		if (info.size() > 0)
			std::cout << "Build log: " << info << std::endl;
	}
}

void Viewport::createShader()
{
	std::string vertex = 
		"void main(void) {\n"
			"gl_Position     = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
			"gl_FrontColor   = gl_Color;\n"
			"gl_TexCoord[0]  = gl_MultiTexCoord0;\n"
		"}\n";

	std::string fragment =
		"void main(void) {\n"
			"vec2 coords = vec2(gl_TexCoord[0]);\n"
			"gl_FragColor = vec4(coords.s, coords.t, 1.0 - coords.s + coords.t, 1.0);\n"
		"}\n";

	if (m_waves.size() > 0) {
		fragment =
			"uniform float timer;\n"
			"uniform sampler2D tex1;\n"
			"uniform float xoffs;\n"
			"uniform float yoffs;\n"
			"void main(void) {\n"
				"vec2 TexCoord = vec2(gl_TexCoord[0]) + vec2(xoffs, yoffs);\n"
				"float amp = 0.0;\n"
				"float dist;\n";

		for (int i = 0; i < m_waves.size(); ++i) {
			fragment += "dist = length(TexCoord - texture2D(tex1, vec2(";
			fragment += util::toString(i);
			fragment += ".0 / 32.0 + 0.5 / 32.0, 0.0)).xy);\n";
			fragment += "amp += 637.5 / ";
			fragment += util::toString(m_waves.size());
			fragment += ".0 * sin(6.2831 * (timer / 3.0 - dist / 0.2));\n";
		}

		fragment +=
			"amp = 2.0 * abs(amp);\n"
			"float r = 0.0;\n"
			"float g = 0.0;\n"
			"float b = 0.0;\n"
			"if (amp <= 255.0) {\n"
				"r = 1.0;\n"
				"g = amp / 255.0;\n"
			"} else if (amp <= 510.0) {\n"
				"r = (-amp + 500.0) / 255.0;\n"
				"g = 1.0;\n"
			"} else if (amp <= 765.0) {\n"
				"g = 1.0;\n"
				"b = (amp - 510.0) / 255.0;\n"
			"} else if (amp <= 1020.0) {\n"
				"g = (-amp + 1020.0) / 255.0;\n"
				"b = 1.0;\n"
			"} else {\n"
				"r = (amp - 1024.0) / 255.0;\n"
				"b = 1.0;\n"
			"}\n"
			"gl_FragColor = (1.0 - vec4(r, g, b, 0.0));\n"
			//"gl_FragColor = vec4(texture2D(tex1, TexCoord));\n"
		"}";
	}

	//std::cout << fragment << std::endl;

	m_shader = ogl::__Shader::loadFromSource(vertex.c_str(), fragment.c_str());

	m_shader->compile();
	m_shader->bind();
}

void Viewport::mouseDoubleClick(util::Button button, int x, int y)
{
}

void Viewport::mouseWheel(int delta)
{
}



void Viewport::keyPressEvent(QKeyEvent* event)
{
	m_keyAdapter.keyEvent(event);
}

void Viewport::keyReleaseEvent(QKeyEvent* event)
{
	m_keyAdapter.keyEvent(event);
}

void Viewport::mouseMoveEvent(QMouseEvent* event)
{
	m_mouseAdapter.mouseEvent(event);
}

void Viewport::mousePressEvent(QMouseEvent* event)
{
	m_mouseAdapter.mouseEvent(event);
}

void Viewport::mouseReleaseEvent(QMouseEvent* event)
{
	m_mouseAdapter.mouseEvent(event);
}

void Viewport::wheelEvent(QWheelEvent* event)
{
	m_mouseAdapter.mouseWheelEvent(event);
}

void Viewport::mouseDoubleClickEvent(QMouseEvent* event)
{
	m_mouseAdapter.mouseEvent(event);
}
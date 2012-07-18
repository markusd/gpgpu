#ifndef WAVES_HPP_
#define WAVES_HPP_

#include <m3d/m3d.hpp>

#define __CL_ENABLE_EXCEPTIONS
// use float instead of unsigned byte for CPU, for real
#define CPU_FLOAT

// Use OpenCL<->OpenGL interop for direct texture writes from kernels
#define GL_INTEROP

// Use a render buffer as target for the kernel
//#define GL_FBO

// Use CUDA implementation
#define USE_CUDA

// Use CUDA<->OpenGL interop for direct texture writes from kernels
//#define CUDA_GL_INTEROP

// Start the DirectCompute version. You have to set the Linker to Windows, not Console
//#define USE_DIRECT_COMPUTE

//#define USE_D3D11

#define WIDTH 32
#define HEIGHT 32

enum ComputationMode {
	CPU,
	OPENCL,
#ifdef USE_CUDA
	CUDA,
#endif
	SHADER
};

using namespace m3d;

struct Wave {
	Vec2f pos;

	Wave(float x, float y) {
		pos = Vec2f(x, y);
	}
};

#endif
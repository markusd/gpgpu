#ifndef WAVES_HPP_
#define WAVES_HPP_

#include <m3d/m3d.hpp>

#define __CL_ENABLE_EXCEPTIONS

enum ComputationMode {
	CPU,
	OPENCL,
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
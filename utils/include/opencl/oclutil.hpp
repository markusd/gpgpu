#ifndef OCL_UTIL_HPP_
#define OCL_UTIL_HPP_

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <string>

namespace ocl {


bool createContext(cl_device_type type, cl_context_properties glContext, cl_context_properties hDC,
	cl::Platform& platform, std::vector<cl::Device>& devices, cl::Context& context, cl::CommandQueue& queue);

bool createContextEx(cl_device_type type,
					 cl::Platform& platform,
					 std::vector<cl::Device>& devices,
					 cl::Context& context,
					 std::vector<cl::CommandQueue>& queues);

int roundGlobalSize(int groupSize, int globalSize);

}

#endif
#ifndef OCL_UTIL_HPP_
#define OCL_UTIL_HPP_

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <string>

namespace ocl {

	/*
	bool createContext(cl_device_type devType,
		cl_context_properties glContext, cl_context_properties hDC,
		cl_platform_id* platform, cl_device_id* device, cl_context* context, cl_command_queue* queue);

	bool createProgram(cl_context context, std::string source, cl_program* program);
	bool createKernel(cl_program program, std::string name, cl_kernel* kernel);

	cl_device_id getFirstDevice(cl_context context);
*/

bool createContext(cl_device_type type, cl_context_properties glContext, cl_context_properties hDC,
	cl::Platform& platform, std::vector<cl::Device>& devices, cl::Context& context, cl::CommandQueue& queue);

int roundGlobalSize(int groupSize, int globalSize);

}

#endif
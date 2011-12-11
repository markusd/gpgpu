#include <opencl/oclutil.hpp>
#include <iostream>

namespace ocl {

bool createContext(cl_device_type type, cl_context_properties glContext, cl_context_properties hDC,
	cl::Platform& platform, std::vector<cl::Device>& devices, cl::Context& context, cl::CommandQueue& queue)
{
	cl_int clError = CL_SUCCESS;
	std::string info("");
	platform = NULL;
	try {
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "OpenCL: No platform found" << std::endl;
			return false;
		}
		int use = 0;
		for (int i = 0; i < platforms.size(); ++i) {
			platforms[i].getInfo(CL_PLATFORM_NAME, &info);
			std::cout << "Platform found " << info << ". Press '1' to use or '0' to see the next. " << std::endl;
			platform = platforms[i];
			std::cin >> use;
			std::cin.get();
			std::cout << use << std::endl;
			if (use) break;
		}
		if (platform() == NULL) platform = platforms[0];
		platform.getInfo(CL_PLATFORM_NAME, &info);
		std::cout << "Platform name " << info << std::endl;

		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		if (devices.size() == 0) {
			std::cout << "OpenCL: No device found" << std::endl;
			return false;
		}

		while (devices.size() > 1)
			devices.pop_back();

		for (std::vector<cl::Device>::iterator itr = devices.begin(); itr != devices.end(); ++itr) {
			itr->getInfo(CL_DEVICE_NAME, &info);
			std::cout << "Device Name " << info << std::endl;
		}

		if (glContext && hDC) {
			cl_context_properties props[] = {
				CL_GL_CONTEXT_KHR, (cl_context_properties)glContext, 
				CL_WGL_HDC_KHR, (cl_context_properties)hDC, 
				CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 
				0
			};

			context = cl::Context(CL_DEVICE_TYPE_ALL, props);
		} else {
			cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0 };
			context = cl::Context(CL_DEVICE_TYPE_ALL, props);
		}

		queue = cl::CommandQueue(context, devices[0], 0, NULL);
	} catch (cl::Error clError) {
		std::cerr 
			<< "OpenCL Error: "
			<< clError.what()
			<< "("
			<< clError.err()
			<< ")"
			<< std::endl;
		return false;
	}
	return true;
}

int roundGlobalSize(int groupSize, int globalSize)
{
    int r = globalSize % groupSize;
    if(r == 0) 
        return globalSize;
	else
        return globalSize + groupSize - r;
}

}
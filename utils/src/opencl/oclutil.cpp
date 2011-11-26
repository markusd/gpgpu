#include <opencl/oclutil.hpp>
#include <iostream>
namespace ocl {


/*
bool createContext(cl_device_type devType,
				   cl_context_properties glContext, cl_context_properties hDC,
				   cl_platform_id* platform, cl_device_id* device, cl_context* context, cl_command_queue* queue)
{
	char buffer[1024];
	cl_uint numPlatforms, numDevices; 
	cl_platform_id* platformIDs;
	cl_int error = CL_SUCCESS;
	*platform = NULL;
	*context = NULL;

    error = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (error != CL_SUCCESS) {
		std::cout << "OpenCL: Could not get platforms" << std::endl;
		return false;
	} else if (numPlatforms == 0) {
		std::cout << "OpenCL: No platforms available" << std::endl;
		return false;
	}

	platformIDs = new cl_platform_id[numPlatforms];
	error = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	*platform = platformIDs[0];
	delete[] platformIDs;

	clGetPlatformInfo(*platform, CL_PLATFORM_NAME, sizeof(buffer), &buffer, NULL);
	std::cout << "OpenCL: Platform name " << buffer << std::endl;

	error = clGetDeviceIDs(*platform, devType, 0, NULL, &numDevices);
	if (numDevices == 0) {
		std::cout << "OpenCL: No devices available" << std::endl;
		return false;
	}

	cl_device_id* devices = new cl_device_id[numDevices];
    error = clGetDeviceIDs(*platform, devType, numDevices, devices, NULL);
	*device = devices[0];
	delete[] devices;

	clGetDeviceInfo(*device, CL_DEVICE_NAME, sizeof(buffer), &buffer, NULL);
	std::cout << "OpenCL: Device name " << buffer << std::endl;

	if (glContext && hDC) {
		cl_context_properties props[] = {
			CL_GL_CONTEXT_KHR, (cl_context_properties)glContext, 
			CL_WGL_HDC_KHR, (cl_context_properties)hDC, 
			CL_CONTEXT_PLATFORM, (cl_context_properties)*platform, 
			0
		};
		*context = clCreateContext(props, 1, device, NULL, NULL, &error);
	} else {
		cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)*platform, 0 };
		*context = clCreateContext(props, 1, device, NULL, NULL, &error);
	}
	
	*queue = clCreateCommandQueue(*context, *device, 0, &error);

	return true;
}


bool createProgram(cl_context context, std::string source, cl_program* program)
{
	const char* raw = source.c_str();
	cl_uint size = source.size();
	cl_int error = CL_SUCCESS;

	*program = clCreateProgramWithSource(context, 1, (const char **)&raw, &size, &error);

	error = clBuildProgram(*program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
	if (error != CL_SUCCESS) {
		char buffer[10240];
		clGetProgramBuildInfo(*program, getFirstDevice(context), CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		std::cout << "OpenCL: " << buffer << std::endl;
		return false;
	}

	return true;
}

bool createKernel(cl_program program, std::string name, cl_kernel* kernel)
{
	cl_int error = CL_SUCCESS;
	*kernel = clCreateKernel(program, name.c_str(), &error);
	return true;
}

cl_device_id getFirstDevice(cl_context context)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*)malloc(szParmDataBytes);

    clGetContextInfo(context, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id first = cdDevices[0];
    free(cdDevices);

    return first;
}

*/



bool createContext(cl_device_type type, cl_context_properties glContext, cl_context_properties hDC,
	cl::Platform& platform, std::vector<cl::Device>& devices, cl::Context& context, cl::CommandQueue& queue)
{
	cl_int clError = CL_SUCCESS;
	std::string info("");
	try {
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "OpenCL: No platform found" << std::endl;
			return false;
		}
		for (int i = 0; i < platforms.size(); ++i) {
			std::cout << platforms[i].getInfo(CL_PLATFORM_NAME, &info);
			std::cout << "Platform found " << info << std::endl;
		}
		platform = platforms[0];
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
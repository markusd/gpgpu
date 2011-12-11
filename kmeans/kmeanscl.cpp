#include "kmeans.hpp"

#ifndef USE_VISUALIZATION

#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>

#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include "windows.h"
#undef max
#undef min
#else
#endif

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include <CL/cl.hpp>
#include <opencl/oclutil.hpp>

#include <util/clock.hpp>

const int DIM = 1024;
const int N = 2048;
const int K = 10;
typedef Vec<DIM, float> Vecf;

cl::Platform clPlatform;
std::vector<cl::Device> clDevices;
cl::Context clContext;
cl::CommandQueue clQueue;
cl::Program clProgram;
cl::Kernel clClusterAssignment;
cl::Kernel clComputeMean;
cl::Buffer clInputBuf;
cl::Buffer clCentroidBuf;
cl::Buffer clMappingBuf;
cl::Buffer clMeanBuf;

boost::mt19937 rng;
boost::uniform_real<float> u;
boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >* gen;

float gen_random_float(float min, float max)
{
    return (*gen)();
}

float absf(float a)
{
	return a >= 0.0f ? a : -a;
}

void compute_mean(Vecf* input, int dim, int n, Vecf* mean)
{
	memset((void*)mean, 0, n * sizeof(float));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < dim; ++j) {
			(*mean)[j] += input[i][j];
		}
	}

	for (int j = 0; j < dim; ++j) {
		(*mean)[j] /= (float)n;
	}
}

void cluster_assignment(Vecf* input, int dim, int n, Vecf* centroids, int k, int* mapping)
{
	// for each input vector
	for (int i = 0; i < n; ++i) {
		float min_dist = std::numeric_limits<float>::max();

		// for each centroid
		for (int j = 0; j < k; j++) {
			float dist = (input[i] - centroids[j]).lenlen();
			if (dist < min_dist) {
				mapping[i] = j;
				min_dist = dist;
			}
		}
	}
}

void initCL()
{
	cl_int clError = CL_SUCCESS;
	ocl::createContext(CL_DEVICE_TYPE_ALL, NULL, NULL, clPlatform, clDevices, clContext, clQueue);

	std::ifstream t("kmeans.cl");
	std::string code((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

	try {
		cl::Program::Sources source(1, std::make_pair(code.c_str(), code.size()));
		clProgram = cl::Program(clContext, source);
		clProgram.build(clDevices, "-cl-fast-relaxed-math");

		std::string info("");
		clProgram.getBuildInfo(clDevices[0], CL_PROGRAM_BUILD_LOG, &info);
		if (info.size() > 0)
			std::cout << "Build log: " << info << std::endl;

		clClusterAssignment = cl::Kernel(clProgram, "cluster_assignment", &clError);
		clComputeMean = cl::Kernel(clProgram, "compute_mean", &clError);

	} catch (const cl::Error& err) {
		std::cout << "OpenCL Error 4: " << err.what() << " (" << err.err() << ")" << std::endl;
		std::string info("");
		clProgram.getBuildInfo(clDevices[0], CL_PROGRAM_BUILD_LOG, &info);
		if (info.size() > 0)
			std::cout << "Build log: " << info << std::endl;
		std::cin.get();
	}
}

int main(int argc, char** argv)
{
	float time = 0.0f;
	cl_int clError = CL_SUCCESS;

	rng.seed(GetTickCount());
	u = boost::uniform_real<float>(0.0f, 1.0f);
	gen = new boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >(rng, u);

	initCL();

	Vecf* input = new Vecf[N];
	int* mapping = new int[N];
	int* mapping_ = new int[N];
	Vecf* mean = new Vecf();
	Vecf* mean_ = new Vecf();

	// initialize input
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < DIM; ++j) {
			input[i][j] = gen_random_float(0.0f, 1.0f);
		}
	}

	// first k
	Vecf* centroids = input;

	util::Clock clock;
	clock.reset();

	clInputBuf = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * DIM * sizeof(float), input, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clCentroidBuf = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, K * DIM * sizeof(float), centroids, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clMappingBuf = cl::Buffer(clContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, N * sizeof(int), mapping, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clMeanBuf = cl::Buffer(clContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, N * sizeof(int), mean, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clClusterAssignment.setArgs(clInputBuf(), DIM, N, clCentroidBuf(), K, clMappingBuf());
	clComputeMean.setArgs(clInputBuf(), DIM, N, clMeanBuf());

	clock.reset();

	clQueue.enqueueWriteBuffer(clInputBuf, CL_FALSE, 0, N * DIM * sizeof(float), (void*)input, NULL, NULL);
	clQueue.enqueueWriteBuffer(clCentroidBuf, CL_FALSE, 0, K * DIM * sizeof(float), (void*)centroids, NULL, NULL); 

	clQueue.enqueueNDRangeKernel(clComputeMean, cl::NullRange, cl::NDRange(DIM), cl::NDRange(16), NULL, NULL);
	
	clQueue.enqueueNDRangeKernel(clClusterAssignment, cl::NullRange, 
		cl::NDRange(N), cl::NDRange(16), NULL, NULL);

	clQueue.enqueueReadBuffer(clMeanBuf, CL_FALSE, 0, DIM * sizeof(float), mean, NULL, NULL);
	clQueue.enqueueReadBuffer(clMappingBuf, CL_TRUE, 0, N * sizeof(int), mapping, NULL, NULL);

	time = clock.get();
	std::cout << time << std::endl;
	clock.reset();

	cluster_assignment(input, DIM, N, centroids, K, mapping_);
	//compute_mean(input, DIM, N, mean_);

	time = clock.get();
	std::cout << time << std::endl;
	clock.reset();

	for (int i = 0; i < N; ++i) {
		//std::cout << i << " --> " << mapping[i] << std::endl;
		if (mapping[i] != mapping_[i]) {
			std::cout << "Error mapping" << std::endl;
			break;
		}
	}

	for (int i = 0; i < DIM; ++i) {
		if (absf((*mean)[i] - (*mean_)[i]) > 0.1f) {
			std::cout << "Error mean" << std::endl;
		}
	}
	

	delete[] input;
	delete[] mapping;


	std::cin.get();

	return 0;
}



#endif /* USE_VISUALIZATION */
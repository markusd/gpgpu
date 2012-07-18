#include "kmeans.hpp"


#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>

#include <math.h>


#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include <CL/cl.hpp>
#include <opencl/oclutil.hpp>

#include <util/tostring.hpp>
#include <util/clock.hpp>

#include <m3d/vec.hpp>

typedef Vec<DIM, float> Vecf;

Vecf* input = NULL;
int* mapping = NULL;
Vecf* centroids = NULL;


cl::Platform clPlatform;
std::vector<cl::Device> clDevices;
cl::Context clContext;
std::vector<cl::CommandQueue> clQueues;
cl::Program clProgram;
cl::Kernel clClusterAssignment;
cl::Kernel clComputeMean;
cl::Kernel clClusterReposition;
cl::Kernel clClusterReposition_k;
cl::Kernel clClusterReposition_k_c;
cl::Buffer clInputBuf;
cl::Buffer clCentroidBuf;
//cl::Buffer clNewCentroidBuf;
cl::Buffer clMappingBuf;
//cl::Buffer clMeanBuf;
cl::Buffer clConvergedBuf;


boost::mt19937 rng;
boost::uniform_real<float> u;
boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >* gen;

float gen_random_float()
{
    return (*gen)();
}

void initCL()
{
	ocl::createContextEx(CL_DEVICE_TYPE_ALL, clPlatform, clDevices, clContext, clQueues);
	cl_int clError = CL_SUCCESS;

	std::ifstream t("kmeans.cl");
	std::string code((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

	std::string header = "#define DIM ";
	header +=  util::toString(DIM);
	header += "\n";
	header += "#define K ";
	header += util::toString(K);
	header += "\n";
	header += "#define N ";
	header += util::toString(N);
	header += "\n";
	header += "#define AM_LWS ";
	header += util::toString(AM_LWS);
	header += "\n";
	header += "#define RP_LWS ";
	header += util::toString(RP_LWS);
	header += "\n\n\n";

	code = header + code;

	try {
		cl::Program::Sources source(1, std::make_pair(code.c_str(), code.size()));
		clProgram = cl::Program(clContext, source);
		clProgram.build(clDevices, "-cl-fast-relaxed-math -cl-unsafe-math-optimizations -cl-mad-enable");

		std::string info("");
		for (std::vector<cl::Device>::iterator itr = clDevices.begin(); itr != clDevices.end(); ++itr) {
			clProgram.getBuildInfo(*itr, CL_PROGRAM_BUILD_LOG, &info);
			if (info.size() > 0)
				std::cout << "Build log: " << info << std::endl;
		}

		clClusterAssignment = cl::Kernel(clProgram, "cluster_assignment", &clError);
		clComputeMean = cl::Kernel(clProgram, "compute_mean", &clError);
		clClusterReposition = cl::Kernel(clProgram, "cluster_reposition", &clError);
		clClusterReposition_k = cl::Kernel(clProgram, "cluster_reposition_k", &clError);
		clClusterReposition_k_c = cl::Kernel(clProgram, "c_cluster_reposition", &clError);

	} catch (const cl::Error& err) {
		std::cout << "OpenCL Error 4: " << err.what() << " (" << err.err() << ")" << std::endl;
		std::string info("");
		for (std::vector<cl::Device>::iterator itr = clDevices.begin(); itr != clDevices.end(); ++itr) {
			clProgram.getBuildInfo(*itr, CL_PROGRAM_BUILD_LOG, &info);
			if (info.size() > 0)
				std::cout << "Build log: " << info << std::endl;
		}
		std::cin.get();
	}
}

void exec(int id)
{
	std::cout << "thread:" << boost::this_thread::get_id() << ": " << id << std::endl;

	const cl::CommandQueue& queue = clQueues[id];

	util::Clock clock;
	clock.reset();

	int* tmapping = new int[N];
	Vecf* tcentroids = new Vecf[K];



	queue.enqueueWriteBuffer(clInputBuf, CL_FALSE, 0, N * DIM * sizeof(float), (void*)input, NULL, NULL);
	queue.enqueueWriteBuffer(clCentroidBuf, CL_FALSE, 0, K * DIM * sizeof(float), (void*)centroids, NULL, NULL);



	for (int i = 0; i < ITERATIONS; ++i) {
		queue.enqueueNDRangeKernel(clClusterAssignment, cl::NullRange, cl::NDRange(N), cl::NDRange(AM_LWS), NULL, NULL);

#ifdef CLUSTER_REPOSITION_K
		queue.enqueueNDRangeKernel(clClusterReposition_k, cl::NullRange, cl::NDRange(K), cl::NDRange(RP_LWS), NULL, NULL);
#else
		queue.enqueueNDRangeKernel(clClusterReposition, cl::NullRange, cl::NDRange(DIM), cl::NDRange(RP_LWS), NULL, NULL);
#endif
	}

	queue.finish();

	queue.enqueueReadBuffer(clCentroidBuf, CL_FALSE, 0, K * DIM * sizeof(float), tcentroids, NULL, NULL);
	queue.enqueueReadBuffer(clMappingBuf, CL_FALSE, 0, N * sizeof(int), tmapping, NULL, NULL);

	queue.finish();

	float now = clock.get();
	std::cout << "Device: " << now << std::endl;
}

int main(int argc, char** argv)
{
	float time = 0.0f;

	rng.seed(GetTickCount());
	u = boost::uniform_real<float>(0.0f, 10.0f);
	gen = new boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >(rng, u);


	cl_int clError = CL_SUCCESS;
	initCL();

	input = new Vecf[N];
	mapping = new int[N];

	// initialize input
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < DIM; ++j) {
			input[i][j] = gen_random_float();
		}
	}

	// first k
	centroids = new Vecf[K];
	for (int i = 0; i < K; ++i)
		memcpy((void*)centroids[i].v, (void*)input[i].v, DIM * sizeof(float));



	clInputBuf = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * DIM * sizeof(float), input, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clCentroidBuf = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, K * DIM * sizeof(float), centroids, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clMappingBuf = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(int), mapping, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clClusterAssignment.setArgs(clInputBuf(), clCentroidBuf(), clMappingBuf());
	clClusterReposition.setArgs(clInputBuf(), clMappingBuf(), clCentroidBuf());
	clClusterReposition_k.setArgs(clInputBuf(), clMappingBuf(), clCentroidBuf());
	clClusterReposition_k_c.setArgs(clInputBuf(), clMappingBuf(), clCentroidBuf(), clConvergedBuf());

	util::Clock clock;
	clock.reset();


	boost::thread_group threads;

	for (int i = 0; i < clQueues.size(); ++i) {
		threads.create_thread(boost::bind(exec, i));
	}

	threads.join_all();

	float now = clock.get();
	std::cout << "Total: " << now << std::endl;


	delete[] input;
	delete[] mapping;

	system("pause");
	std::cin.get();

	return 0;
}
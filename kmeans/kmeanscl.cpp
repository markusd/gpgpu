#include "kmeans.hpp"

#ifndef USE_VISUALIZATION
#ifndef USE_KMEANS_IMG

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

#include <util/tostring.hpp>
#include <util/clock.hpp>

const int DIM = 1024;
const int N = 4096;
const int K = 10;
typedef Vec<DIM, float> Vecf;

cl::Platform clPlatform;
std::vector<cl::Device> clDevices;
cl::Context clContext;
cl::CommandQueue clQueue;
cl::Program clProgram;
cl::Kernel clClusterAssignment;
cl::Kernel clComputeMean;
cl::Kernel clClusterReposition;
cl::Buffer clInputBuf;
cl::Buffer clCentroidBuf;
cl::Buffer clNewCentroidBuf;
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

void compute_mean(Vecf* input, Vecf* mean)
{
	memset((void*)mean, 0, N * sizeof(float));
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < DIM; ++j) {
			(*mean)[j] += input[i][j];
		}
	}

	for (int j = 0; j < DIM; ++j) {
		(*mean)[j] /= (float)N;
	}
}

void cluster_assignment(Vecf* input, Vecf* centroids, int* mapping)
{
	// for each input vector
	for (int i = 0; i < N; ++i) {
		float min_dist = std::numeric_limits<float>::max();

		// for each centroid
		for (int j = 0; j < K; j++) {
			float dist = (input[i] - centroids[j]).lenlen();

			//for (int l = 0; l < DIM; ++l) {
			//	dist += (input[i][l] - centroids[j][l]) * (input[i][l] - centroids[j][l]);
			//}
			//dist = sqrtf(dist);

			if (dist < min_dist) {
				mapping[i] = j;
				min_dist = dist;
			}
		}
	}
}

void cluster_reposition(Vecf* input, Vecf* centroids, int* mapping)
{
	float count[K];

	for (int i = 0; i < K; ++i) {
		count[i] = 0.0f;
		for (int j = 0; j < DIM; ++j)
			centroids[i][j] = 0.0f;
	}

	for (int i = 0; i < N; ++i) {
		count[mapping[i]] += 1.0f;
		for (int j = 0; j < DIM; ++j) {
			centroids[mapping[i]][j] += input[i][j];
		}
	}

	for (int i = 0; i < K; ++i)
		for (int j = 0; j < DIM; ++j)
			centroids[i][j] /= count[i];
}

void initCL()
{
	cl_int clError = CL_SUCCESS;
	ocl::createContext(CL_DEVICE_TYPE_ALL, NULL, NULL, clPlatform, clDevices, clContext, clQueue);

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
	header += "\n\n\n";

	code = header + code;

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
		clClusterReposition = cl::Kernel(clProgram, "cluster_reposition", &clError);

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

	std::vector<Vec2d> vinput;
	std::vector<Vec2d> vseed;


	// initialize input
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < DIM; ++j) {
			input[i][j] = gen_random_float(0.0f, 1.0f);
		}
		vinput.push_back(Vec2d(input[i][0], input[i][1]));
		if (i < K)
			vseed.push_back(Vec2d(input[i][0], input[i][1]));
	}

	

	//std::pair<std::vector<Vec2d>, double> vres = kmeans(10, K, vinput, vseed);

	// first k
	Vecf* centroids = new Vecf[K];
	for (int i = 0; i < K; ++i)
		memcpy((void*)centroids[i].v, (void*)input[i].v, DIM * sizeof(float));

	Vecf* centroids_ = new Vecf[K];
	for (int i = 0; i < K; ++i)
		memcpy((void*)centroids_[i].v, (void*)input[i].v, DIM * sizeof(float));

	util::Clock clock;
	clock.reset();

	clInputBuf = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * DIM * sizeof(float), input, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clCentroidBuf = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, K * DIM * sizeof(float), centroids, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	//clNewCentroidBuf = cl::Buffer(clContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, K * DIM * sizeof(float), centroids, &clError);
	//if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clMappingBuf = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(int), mapping, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clMeanBuf = cl::Buffer(clContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, N * sizeof(int), mean, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clClusterAssignment.setArgs(clInputBuf(), clCentroidBuf(), clMappingBuf());
	clComputeMean.setArgs(clInputBuf(), clMeanBuf());
	clClusterReposition.setArgs(clInputBuf(), clMappingBuf(), clCentroidBuf());

	clock.reset();

	clQueue.enqueueWriteBuffer(clInputBuf, CL_FALSE, 0, N * DIM * sizeof(float), (void*)input, NULL, NULL);
	clQueue.enqueueWriteBuffer(clCentroidBuf, CL_TRUE, 0, K * DIM * sizeof(float), (void*)centroids, NULL, NULL); 

	time = clock.get();
	std::cout << time << std::endl;

	//clQueue.enqueueNDRangeKernel(clComputeMean, cl::NullRange, cl::NDRange(DIM), cl::NDRange(16), NULL, NULL);
	
	for (int i = 0; i < 20; ++i) {
		clQueue.enqueueNDRangeKernel(clClusterAssignment, cl::NullRange, 
			cl::NDRange(N), cl::NDRange(16), NULL, NULL);

		clQueue.enqueueNDRangeKernel(clClusterReposition, cl::NullRange, cl::NDRange(DIM), cl::NDRange(16), NULL, NULL);
	}

	//clQueue.enqueueReadBuffer(clMeanBuf, CL_FALSE, 0, DIM * sizeof(float), mean, NULL, NULL);
	clQueue.enqueueReadBuffer(clCentroidBuf, CL_TRUE, 0, K * DIM * sizeof(float), centroids, NULL, NULL);
	clQueue.enqueueReadBuffer(clMappingBuf, CL_TRUE, 0, N * sizeof(int), mapping, NULL, NULL);

	time = clock.get();
	std::cout << time << std::endl;
	clock.reset();


	for (int i = 0; i < 20; ++i) {
		cluster_assignment(input, centroids_, mapping_);
		cluster_reposition(input, centroids_, mapping_);

		//for (int j = 0; j < K; ++j)
		//	std::cout << centroids[j][0] << ", " << centroids[j][1] << std::endl;

		//std::cout << std::endl;
	}
	//compute_mean(input, mean_);

	float ctime = clock.get();
	std::cout << ctime << std::endl;
	clock.reset();

	std::cout << ctime / time << " x speedup" << std::endl;

	for (int i = 0; i < N; ++i) {
		//std::cout << i << " --> " << mapping[i] << std::endl;
		if (mapping[i] != mapping_[i]) {
			std::cout << "Error mapping" << std::endl;
			break;
		}
	}

	for (int i = 0; i < DIM; ++i) {
		if (absf((*mean)[i] - (*mean_)[i]) > 0.001f) {
			std::cout << "Error mean" << std::endl;
			break;
		}
	}

	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < DIM; ++j) {
			if (absf(centroids[i][j] - centroids_[i][j]) > 0.001f) {
				std::cout << "Error centroid" << std::endl;
				break;
			}
		}
	}
/*
	for (int i = 0; i < K; ++i) {
		std::cout << centroids_[i][0] << " . " <<  vres.first[i].x <<
			"   " << centroids_[i][1] << " . " <<  vres.first[i].y << std::endl;
	}
*/	

	delete[] input;
	delete[] mapping;


	std::cin.get();

	return 0;
}


#endif /* USE_KMEANS_IMG */
#endif /* USE_VISUALIZATION */
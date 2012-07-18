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

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <kmeans.cuh>
#else
#include <CL/cl.hpp>
#include <opencl/oclutil.hpp>
#endif

#include <util/tostring.hpp>
#include <util/clock.hpp>

#include <kmeans_def.hpp>
typedef Vec<DIM, float> Vecf;

#ifdef USE_CUDA
#else
cl::Platform clPlatform;
std::vector<cl::Device> clDevices;
cl::Context clContext;
cl::CommandQueue clQueue;
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
#endif


boost::mt19937 rng;
boost::uniform_real<float> u;
boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >* gen;

float gen_random_float()
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

/*
struct ElemDist {

	//ElemDist
	float distance;
	int index;
}

void hartigan_wong(Vecf* input, Vecf* seed)
{
	Vecf mean(&input[0].v[0]);

	for (int i = 1; i < N; ++i) {
		mean += input[i];
	}
	mean *= (1.0f / (float)N);

	ElemDist* distance new ElemDist[N];
	for (int i = 1; i < N; ++i) {
		distance[i] = mean.distsqr(input[i]);
	}


	int* sorted = new int[N];


	std::vector<Vec2d> sorted(input);

	compare_mean = mean;
	std::sort(sorted.begin(), sorted.end(), compare_dist);

	//for (int i = 0; i < sorted.size(); ++i)
	//	std::cout << (sorted[i] - mean).len() << std::endl;
	//std::cout << std::endl;

	unsigned int gap = (N - 1) / std::max<unsigned int>(K - 1, 1);

	for (int i = 0; i < K; ++i)
		for (int j = 0; j < DIM; ++j)
			seed[i][j] = 0.0f;
			//seed.push_back(sorted[i*gap]);

	return std::make_pair(mean, seed);
}
*/

void cluster_assignment(Vecf* input, Vecf* centroids, int* mapping)
{
	// for each input vector
	for (int i = 0; i < N; ++i) {
		float min_dist = std::numeric_limits<float>::max();

		// for each centroid
		for (int j = 0; j < K; j++) {
			float dist = (input[i] - centroids[j]).lenlen();
			//float dist = input[i].distsqr(centroids[j]);
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

#ifdef USE_CUDA
#else
void initCL()
{
	ocl::createContext(CL_DEVICE_TYPE_ALL, NULL, NULL, clPlatform, clDevices, clContext, clQueue);
	cl_int clError = CL_SUCCESS;

		cl_ulong size;
		clDevices[0].getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
		std::cout << size << std::endl;

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
		clProgram.getBuildInfo(clDevices[0], CL_PROGRAM_BUILD_LOG, &info);
		if (info.size() > 0)
			std::cout << "Build log: " << info << std::endl;

		clClusterAssignment = cl::Kernel(clProgram, "cluster_assignment", &clError);
		clComputeMean = cl::Kernel(clProgram, "compute_mean", &clError);
		clClusterReposition = cl::Kernel(clProgram, "cluster_reposition", &clError);
		clClusterReposition_k = cl::Kernel(clProgram, "cluster_reposition_k", &clError);
		clClusterReposition_k_c = cl::Kernel(clProgram, "c_cluster_reposition", &clError);

	} catch (const cl::Error& err) {
		std::cout << "OpenCL Error 4: " << err.what() << " (" << err.err() << ")" << std::endl;
		std::string info("");
		clProgram.getBuildInfo(clDevices[0], CL_PROGRAM_BUILD_LOG, &info);
		if (info.size() > 0)
			std::cout << "Build log: " << info << std::endl;
		std::cin.get();
	}
}
#endif

int main(int argc, char** argv)
{
	float time = 0.0f;

	rng.seed(GetTickCount());
	u = boost::uniform_real<float>(0.0f, 10.0f);
	gen = new boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >(rng, u);

#ifdef USE_CUDA
#else
	cl_int clError = CL_SUCCESS;
	initCL();
#endif

	Vecf* input = new Vecf[N];
	int* mapping = new int[N];
	int* mapping_ = new int[N];
	Vecf* mean = new Vecf();
	Vecf* mean_ = new Vecf();

	int converged = 0;


	// initialize input
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < DIM; ++j) {
			input[i][j] = gen_random_float();
		}
	}

	// first k
	Vecf* centroids = new Vecf[K];
	for (int i = 0; i < K; ++i)
		memcpy((void*)centroids[i].v, (void*)input[i].v, DIM * sizeof(float));

	Vecf* centroids_ = new Vecf[K];
	for (int i = 0; i < K; ++i)
		memcpy((void*)centroids_[i].v, (void*)input[i].v, DIM * sizeof(float));

	util::Clock clock;
	clock.reset();

#ifdef USE_CUDA
	float* cuInput = NULL;
	float* cuCentroid = NULL;
	int* cuMapping = NULL;
	cudaMalloc((void **)&cuInput, N * DIM * sizeof(float));
	cudaMalloc((void **)&cuCentroid, K * DIM * sizeof(float));
	cudaMalloc((void **)&cuMapping, N * sizeof(int));
#else

	clInputBuf = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * DIM * sizeof(float), input, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clCentroidBuf = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, K * DIM * sizeof(float), centroids, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	//clNewCentroidBuf = cl::Buffer(clContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, K * DIM * sizeof(float), centroids, &clError);
	//if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clMappingBuf = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(int), mapping, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	//clMeanBuf = cl::Buffer(clContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, N * sizeof(int), mean, &clError);
	//if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clConvergedBuf = cl::Buffer(clContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int), &converged, &clError);
	if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

	clClusterAssignment.setArgs(clInputBuf(), clCentroidBuf(), clMappingBuf());
	//clComputeMean.setArgs(clInputBuf(), clMeanBuf());
	clClusterReposition.setArgs(clInputBuf(), clMappingBuf(), clCentroidBuf());
	clClusterReposition_k.setArgs(clInputBuf(), clMappingBuf(), clCentroidBuf());
	clClusterReposition_k_c.setArgs(clInputBuf(), clMappingBuf(), clCentroidBuf(), clConvergedBuf());

#endif

#ifdef USE_CUDA
	cudaMemcpy(cuInput, input, N * DIM * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuCentroid, centroids, K * DIM * sizeof(float), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();

	for (int i = 0; i < ITERATIONS; ++i) {
		cuda_launch_assignment(cuInput, cuCentroid, cuMapping);
	}
	cudaThreadSynchronize();
#else
	clQueue.enqueueWriteBuffer(clInputBuf, CL_FALSE, 0, N * DIM * sizeof(float), (void*)input, NULL, NULL);
	clQueue.enqueueWriteBuffer(clCentroidBuf, CL_FALSE, 0, K * DIM * sizeof(float), (void*)centroids, NULL, NULL);
	

	clQueue.finish();
	for (int i = 0; !converged && (i < ITERATIONS); ++i) {

		converged = 1;
		clQueue.enqueueWriteBuffer(clConvergedBuf, CL_FALSE, 0, sizeof(int), (void*)&converged, NULL, NULL);

		std::cout << i << "\t" << clock.get() << std::endl;
		clQueue.enqueueNDRangeKernel(clClusterAssignment, cl::NullRange, 
			cl::NDRange(N), cl::NDRange(AM_LWS), NULL, NULL);
		clQueue.finish();
		std::cout << "\t\t" << clock.get() << std::endl;
#ifdef CLUSTER_REPOSITION_K
		clQueue.enqueueNDRangeKernel(clClusterReposition_k_c, cl::NullRange, cl::NDRange(K), cl::NDRange(RP_LWS), NULL, NULL);
#else
		clQueue.enqueueNDRangeKernel(clClusterReposition, cl::NullRange, cl::NDRange(DIM), cl::NDRange(RP_LWS), NULL, NULL);
#endif
		clQueue.enqueueReadBuffer(clConvergedBuf, CL_FALSE, 0, sizeof(int), &converged, NULL, NULL);
		clQueue.finish();
	}

	clQueue.finish();
#endif

	clock.reset();

#ifdef USE_CUDA
	cudaMemcpy(cuInput, input, N * DIM * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuCentroid, centroids, K * DIM * sizeof(float), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
#else
	clQueue.enqueueWriteBuffer(clInputBuf, CL_FALSE, 0, N * DIM * sizeof(float), (void*)input, NULL, NULL);
	clQueue.enqueueWriteBuffer(clCentroidBuf, CL_FALSE, 0, K * DIM * sizeof(float), (void*)centroids, NULL, NULL);

	clQueue.finish();
#endif

	float afterWrite = clock.get();
	std::cout << "Write: " << afterWrite << "\n";

#ifdef USE_CUDA
	for (int i = 0; i < ITERATIONS; ++i) {
		cuda_launch_assignment(cuInput, cuCentroid, cuMapping);
	}
	cudaThreadSynchronize();
#else
	//clQueue.enqueueNDRangeKernel(clComputeMean, cl::NullRange, cl::NDRange(DIM), cl::NDRange(16), NULL, NULL);
	float _t = clock.get();
	float __t = 0.0f;
	for (int i = 0; i < ITERATIONS; ++i) {
		clQueue.enqueueNDRangeKernel(clClusterAssignment, cl::NullRange, 
			cl::NDRange(N), cl::NDRange(AM_LWS), NULL, NULL);
		clQueue.finish();
		__t = clock.get();
		std::cout << "\t" << __t - _t << "\n";
		
#ifdef CLUSTER_REPOSITION_K
		clQueue.enqueueNDRangeKernel(clClusterReposition_k, cl::NullRange, cl::NDRange(K), cl::NDRange(RP_LWS), NULL, NULL);
#else
		clQueue.enqueueNDRangeKernel(clClusterReposition, cl::NullRange, cl::NDRange(DIM), cl::NDRange(RP_LWS), NULL, NULL);
#endif
		clQueue.finish();
		_t = clock.get();
		std::cout << "\t\t" << _t - __t << "\n";
	}

	clQueue.finish();
#endif

	float afterCompute = clock.get();
	std::cout << "Compute: " << afterCompute - afterWrite << "\n";

#ifdef USE_CUDA
	cudaMemcpy(input, cuInput, N * DIM * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(centroids, cuCentroid,  K * DIM * sizeof(float), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	cudaFree(cuInput);
	cudaFree(cuCentroid);
	cudaFree(cuMapping);
#else
	//clQueue.enqueueReadBuffer(clMeanBuf, CL_FALSE, 0, DIM * sizeof(float), mean, NULL, NULL);
	clQueue.enqueueReadBuffer(clCentroidBuf, CL_FALSE, 0, K * DIM * sizeof(float), centroids, NULL, NULL);
	clQueue.enqueueReadBuffer(clMappingBuf, CL_FALSE, 0, N * sizeof(int), mapping, NULL, NULL);

	clQueue.finish();
#endif

	float afterRead = clock.get();
	std::cout << "Read: " << afterRead - afterCompute << std::endl;
	std::cout << "Total: " << afterRead << std::endl;
	std::cout << "Memory: " << afterWrite + (afterRead - afterCompute) << " : " << (afterWrite + (afterRead - afterCompute)) / afterRead << std::endl;

	clock.reset();


	for (int i = 0; i < ITERATIONS; ++i) {
		cluster_assignment(input, centroids_, mapping_);
		cluster_reposition(input, centroids_, mapping_);

		//for (int j = 0; j < K; ++j)
		//	std::cout << centroids[j][0] << ", " << centroids[j][1] << std::endl;

		//std::cout << std::endl;
	}
	//compute_mean(input, mean_);

	float ctime = clock.get();
	std::cout << "CPU: " << ctime << std::endl;
	clock.reset();

	std::cout << ctime / afterRead << " x speedup" << std::endl;
	std::cout << ctime / (afterCompute - afterWrite) << " x speedup (only compute)" << std::endl;

	for (int i = 0; i < N; ++i) {
		//std::cout << i << " --> " << mapping[i] << std::endl;
		if (mapping[i] != mapping_[i]) {
			std::cout << "Error mapping" << std::endl;
			break;
		}
	}

	for (int i = 0; i < DIM; ++i) {
		if (absf((*mean)[i] - (*mean_)[i]) > 0.01f) {
			std::cout << "Error mean" << std::endl;
			break;
		}
	}

	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < DIM; ++j) {
			if (absf(centroids[i][j] - centroids_[i][j]) > 0.01f) {
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

	system("pause");
	std::cin.get();

	return 0;
}


#endif /* USE_KMEANS_IMG */
#endif /* USE_VISUALIZATION */
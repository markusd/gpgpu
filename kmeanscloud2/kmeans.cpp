#include "kmeans.hpp"

#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>

#include <math.h>


#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#define BOOST_FILESYSTEM_VERSION 2
#include <boost/filesystem.hpp>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#ifdef USE_OPENCL
	#include <CL/cl.hpp>
	#include <opencl/oclutil.hpp>
#else
	#include <limits>
#endif

#include <util/tostring.hpp>
#include <util/clock.hpp>

#include <string.h>

#ifdef _WIN32
	#include <windows.h>
	#undef max
	#undef min
#else
	#include <sys/time.h>
#endif

int DIM = 1024;
int N = 2048;
int K = 16;
int ITERATIONS = 20;
int RUNS = 100;

int AM_LWS = 8;
int RP_LWS = 64;
int CT_LWS = 8;

int USE_ALL_DEVICES = 1;
int device_count = 1;

#ifdef USE_OPENCL
cl::Platform clPlatform;
std::vector<cl::Device> clDevices;
cl::Context clContext;
std::vector<cl::CommandQueue> clQueues;
cl::Program clProgram;
std::vector<cl::Kernel> clClusterAssignment;
std::vector<cl::Kernel> clClusterReposition;
std::vector<cl::Kernel> clClusterReposition_k;
std::vector<cl::Kernel> clClusterReposition_k_c;
std::vector<cl::Kernel> clComputeCost;
std::vector<cl::Buffer> clInputBuf;
std::vector<cl::Buffer> clCentroidBuf;
std::vector<cl::Buffer> clMappingBuf;
std::vector<cl::Buffer> clReductionBuf;
std::vector<cl::Buffer> clConvergedBuf;

boost::thread_group reduction_group;
#endif

std::vector<float*> input_list;
std::vector<float*> centroids_list;
std::vector<int*> mapping_list;
std::vector<float> cost_list;

boost::mt19937 rng;
boost::uniform_real<float> u;
boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >* gen;

#ifndef _WIN32
uint64_t getTimeMs(void)
{
    struct timeval tv;

    gettimeofday(&tv, 0);
    return uint64_t( tv.tv_sec ) * 1000 + tv.tv_usec / 1000;
}
#endif

float gen_random_float()
{
    return (*gen)();
}

#ifdef USE_OPENCL
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

		for (int i = 0; i < clDevices.size(); ++i) {
			clClusterAssignment.push_back(cl::Kernel(clProgram, "cluster_assignment", &clError));
			clClusterReposition.push_back(cl::Kernel(clProgram, "cluster_reposition", &clError));
			clClusterReposition_k.push_back(cl::Kernel(clProgram, "cluster_reposition_k", &clError));
			clClusterReposition_k_c.push_back(cl::Kernel(clProgram, "c_cluster_reposition", &clError));
			clComputeCost.push_back(cl::Kernel(clProgram, "compute_cost", &clError));
		}

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
#endif

float* random_cluster_init(float* input)
{
	int count = 0;
	float* seed = new float[K*DIM];

	// find random, but unique centroids
	while (count < K) {
		float* centroid = &input[(rand() % N) * DIM];
		bool found = false;

		for (int i = 0; i < count; ++i) {
			bool equal = true;
			for (int j = 0; j < DIM; ++j) {
				if (centroid[j] != seed[i*DIM+j]) {
					equal = false;
					break;
				}
			}
			if (equal) {
				found = true;
				break;
			}
		}

		if (!found) {
			memcpy(&seed[(count++)*DIM], centroid, DIM * sizeof(float));
		}
	}

	return seed;
}

#ifndef USE_OPENCL
void cluster_assignment(float* input, float* centroids, int* mapping)
{
	// for each input vector
	for (int i = 0; i < N; ++i) {
		float min_dist = std::numeric_limits<float>::max();
		// for each centroid
		for (int j = 0; j < K; j++) {

			float dist = 0.0f;
			for (int l = 0; l < DIM; ++l) {
				dist += (input[i*DIM+l] - centroids[j*DIM+l]) * (input[i*DIM+l] - centroids[j*DIM+l]);
			}
			dist = sqrtf(dist);

			if (dist < min_dist) {
				mapping[i] = j;
				min_dist = dist;
			}
		}
	}

}

void cluster_reposition(float* input, float* centroids, int* mapping)
{
//	bool converged = true;
	float* count = new float[K];
//	float* old_centroids = new float[K*DIM];
//	memcpy(old_centroids, centroids, K*DIM*sizeof(float));

	for (int i = 0; i < K; ++i) {
		count[i] = 0.0f;
		for (int j = 0; j < DIM; ++j)
			centroids[i*DIM+j] = 0.0f;
	}

	for (int i = 0; i < N; ++i) {
		count[mapping[i]] += 1.0f;
		for (int j = 0; j < DIM; ++j) {
			centroids[mapping[i]*DIM+j] += input[i*DIM+j];
		}
	}

	for (int i = 0; i < K; ++i) {
		//float dist = 0.0f;
		for (int j = 0; j < DIM; ++j) {
			centroids[i*DIM+j] /= count[i];
			//dist += (centroids[i*DIM+j] - old_centroids[i*DIM+j]) * (centroids[i*DIM+j] - old_centroids[i*DIM+j]);
		}
		//if (dist > 0.01f)
		//	converged = false;
	}

	delete[] count;
	//delete[] old_centroids;

	//return converged;
}

float cluster_compute_cost(float* input, float* centroids, int* mapping)
{
	float cost = 0.0f;
	for (int i = 0; i < N; ++i) {
		float dist = 0.0f;
		for (int l = 0; l < DIM; ++l) {
			dist += (input[i*DIM+l] - centroids[mapping[i]*DIM+l]) * (input[i*DIM+l] - centroids[mapping[i]*DIM+l]);
		}
		cost += dist;
	}
	return cost / (float)N;
}
#endif

void reduce_cost(int pass, float* cost)
{
	float sum = 0.0f;
	for (int i = 0; i < N; ++i)
		sum += cost[i];
	cost_list[pass] = sum / (float)N;
	delete[] cost;
}


void exec(int id, bool threaded)
{
	if (threaded)
		std::cout << "thread:" << boost::this_thread::get_id() << ": " << id << std::endl;

	float* input = input_list[id];

#ifdef USE_OPENCL
	const cl::CommandQueue& queue = clQueues[id];
	const cl::Kernel& assignment = clClusterAssignment[id];
	const cl::Kernel& reposition = clClusterReposition[id];
	const cl::Kernel& reposition_k = clClusterReposition_k[id];
	const cl::Kernel& reposition_k_c = clClusterReposition_k_c[id];
	const cl::Kernel& cost = clComputeCost[id];

	const cl::Buffer& inputBuf = clInputBuf[id];
	const cl::Buffer& mappingBuf = clMappingBuf[id];
	const cl::Buffer& centroidBuf = clCentroidBuf[id];
	//const cl::Buffer& convergedBuf = clConvergedBuf[id];
	const cl::Buffer& reductionBuf = clReductionBuf[id];

	queue.enqueueWriteBuffer(inputBuf, CL_FALSE, 0, N * DIM * sizeof(float), (void*)input, NULL, NULL);
#endif

	for (int pass = id; pass < RUNS; pass += device_count) {

		float* reduction = new float[N];
		float* centroids = random_cluster_init(input);
		int* mapping = new int[N];
		centroids_list[pass] = centroids;
		mapping_list[pass] = mapping;

		util::Clock clock;
		clock.reset();

#ifdef USE_OPENCL
		queue.enqueueWriteBuffer(centroidBuf, CL_FALSE, 0, K * DIM * sizeof(float), (void*)centroids, NULL, NULL);

		for (int i = 0; i < ITERATIONS; ++i) {
			queue.enqueueNDRangeKernel(assignment, cl::NullRange, cl::NDRange(N), cl::NDRange(AM_LWS), NULL, NULL);

#ifdef CLUSTER_REPOSITION_K
			queue.enqueueNDRangeKernel(reposition_k, cl::NullRange, cl::NDRange(K), cl::NDRange(RP_LWS), NULL, NULL);
#else
			queue.enqueueNDRangeKernel(reposition, cl::NullRange, cl::NDRange(DIM), cl::NDRange(RP_LWS), NULL, NULL);
#endif
		}
		queue.enqueueNDRangeKernel(cost, cl::NullRange, cl::NDRange(N), cl::NDRange(CT_LWS), NULL, NULL);

		//queue.finish();

		queue.enqueueReadBuffer(centroidBuf, CL_FALSE, 0, K * DIM * sizeof(float), centroids, NULL, NULL);
		queue.enqueueReadBuffer(mappingBuf, CL_FALSE, 0, N * sizeof(int), mapping, NULL, NULL);
		queue.enqueueReadBuffer(reductionBuf, CL_TRUE, 0, N * sizeof(float), reduction, NULL, NULL);

		reduction_group.create_thread(boost::bind(reduce_cost, pass, reduction));


		//queue.finish();
#else
		for (int i = 0; i < ITERATIONS; ++i) {
			cluster_assignment(input, centroids, mapping);
			cluster_reposition(input, centroids, mapping);
			//if (cluster_reposition(input, centroids, mapping)) {
			//	std::cout << "Converged after " << i+1 << " iterations." << std::endl;
			//	break;
			//}
		}

		cost_list[pass] = cluster_compute_cost(input, centroids, mapping);
#endif

		float now = clock.get();
		std::cout << "Device: " << now << std::endl;
	}
/*
	float sum = 0.0f;
	for (int i = 0; i < N; ++i) {
		sum += treduction[i];
	}

	sum /= (float)N;

	std::cout << sum << std::endl;
*/
}

int main(int argc, char** argv)
{
	int args = 1;
#ifdef USE_OPENCL
	if (argc < 10) {
#else
	if (argc < 7) {
#endif
		std::cout << "Not enough arguments" << std::endl;
		system("pause");
		return 1;
	}

	DIM = util::toInt(argv[args++]);
	N = util::toInt(argv[args++]);
	K = util::toInt(argv[args++]);
	ITERATIONS = util::toInt(argv[args++]);
	RUNS = util::toInt(argv[args++]);
#ifdef USE_OPENCL
	AM_LWS = util::toInt(argv[args++]);
	RP_LWS = util::toInt(argv[args++]);
	CT_LWS = util::toInt(argv[args++]);

	USE_ALL_DEVICES = util::toInt(argv[args++]);
#else
	device_count = util::toInt(argv[args++]);
#endif


	std::cout << "DIM = " << DIM << std::endl;
	std::cout << "N = " << N << std::endl;
	std::cout << "K = " << K << std::endl;
	std::cout << "ITERATIONS = " << ITERATIONS << std::endl;
	std::cout << "RUNS = " << RUNS << std::endl;
#ifdef USE_OPENCL
	std::cout << "AM_LWS = " << AM_LWS << std::endl;
	std::cout << "RP_LWS = " << RP_LWS << std::endl;
	std::cout << "CT_LWS = " << CT_LWS << std::endl;
	std::cout << "USE_ALL_DEVICES = " << USE_ALL_DEVICES << std::endl << std::endl;
#else
	std::cout << "device_count = " << device_count << std::endl << std::endl;
#endif


#ifdef _WIN32
	rng.seed();
	srand(GetTickCount());
#else
	rng.seed();
	srand(getTimeMs());
#endif

	u = boost::uniform_real<float>(0.0f, 1000000.0f);
	gen = new boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >(rng, u);

#ifdef USE_OPENCL
	cl_int clError = CL_SUCCESS;
	initCL();

	for (int i = 0; i < clDevices.size(); ++i) {

		clInputBuf.push_back(cl::Buffer(clContext, CL_MEM_READ_ONLY, N * DIM * sizeof(float), NULL, &clError));
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

		clCentroidBuf.push_back(cl::Buffer(clContext, CL_MEM_READ_WRITE, K * DIM * sizeof(float), NULL, &clError));
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

		clMappingBuf.push_back(cl::Buffer(clContext, CL_MEM_READ_WRITE, N * sizeof(int), NULL, &clError));
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

		clReductionBuf.push_back(cl::Buffer(clContext, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &clError));
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

		clClusterAssignment[i].setArgs(clInputBuf[i](), clCentroidBuf[i](), clMappingBuf[i]());
		clClusterReposition[i].setArgs(clInputBuf[i](), clMappingBuf[i](), clCentroidBuf[i]());
		clClusterReposition_k[i].setArgs(clInputBuf[i](), clMappingBuf[i](), clCentroidBuf[i]());
		//clClusterReposition_k_c[i].setArgs(clInputBuf[i](), clMappingBuf[i](), clCentroidBuf[i](), clConvergedBuf[i]());
		clComputeCost[i].setArgs(clInputBuf[i](), clCentroidBuf[i](), clMappingBuf[i](), clReductionBuf[i]());

	}

	device_count = clDevices.size();
#endif

	util::Clock clock;
	clock.reset();

	for (int i = 0; i < RUNS; ++i) {
		mapping_list.push_back(NULL);
		centroids_list.push_back(NULL);
		cost_list.push_back(0.0f);
	}

	float* source = new float[N*DIM];
	for (int i = 0; i < N*DIM; ++i)
		source[i] = gen_random_float();

	input_list.push_back(source);

	for (int i = 1; i < device_count; ++i) {
		float* copy = new float[N*DIM];
		memcpy(copy, source, N*DIM*sizeof(float));
		input_list.push_back(copy);
	}

	if (device_count > 1) {
		boost::thread_group threads;

		for (int i = 0; i < device_count; ++i) {
			threads.create_thread(boost::bind(exec, i, true));
		}

		threads.join_all();
	} else {
		exec(0, false);
	}

#ifdef USE_OPENCL
	reduction_group.join_all();
#endif

	int best_result = 0;
	float best_cost = std::numeric_limits<float>::max();
	for (int i = 0; i < RUNS; ++i) {
		if (cost_list[i] < best_cost) {
			best_cost = cost_list[i];
			best_result = i;
		}
	}

	FILE *out_fdesc = fopen("centroids.out", "wb");
	fwrite((void*)centroids_list[best_result], K * DIM * sizeof(float), 1, out_fdesc);
	fclose(out_fdesc);

	out_fdesc = fopen("mapping.out", "wb");
	fwrite((void*)mapping_list[best_result], N * sizeof(int), 1, out_fdesc);
	fclose(out_fdesc);

	std::cout << "Best result is " << best_result << std::endl;

	for (int i = 0; i < device_count; ++i) {
		delete[] input_list[i];
	}

	for (int i = 0; i < RUNS; ++i) {
		delete[] mapping_list[i];
		delete[] centroids_list[i];
	}

	float now = clock.get();
	std::cout << "Total: " << now << std::endl;

	system("pause");

	return 0;
}

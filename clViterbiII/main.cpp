//#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <streambuf>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include <CL/cl.hpp>

boost::mt19937 rng;
boost::uniform_real<float> u;
boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >* gen;

boost::mt19937 rngd;
boost::uniform_real<double> ud;
boost::variate_generator<boost::mt19937&, boost::uniform_real<double> >* gend;

#define FLOAT_TYPE cl_double
#define REAL_MAX DBL_MAX

//#pragma pack(4)
typedef struct vfield {
	FLOAT_TYPE probability;
	cl_int state;
	} vfield_t;

// HMM
cl_int ObservableStates;
cl_int HiddenStates;
cl_int nObservations;
cl_int* Observations;		// vector
FLOAT_TYPE* StartProbability = NULL;	// vector
FLOAT_TYPE* TransitionMatrix = NULL;	// matrix[i][j] i-->j matrix[(i*HiddenStates) + j] 
FLOAT_TYPE* EmissionProbability = NULL;// matrix[i][j] i-->j matrix[(i*ObservableStates) + j]
// TODO: optimize TransitionMatrix with coalesced access

cl_int rc;

// GPU
vfield_t* V = NULL; // V[i][t] = V[i + (t*HiddenStates)]
cl_int* Path = NULL; // vector
FLOAT_TYPE PathProbability = 0.0f;
// CPU
vfield_t* Vc = NULL;
cl_int* PathC = NULL;
FLOAT_TYPE PathProbabilityC = 0.0f;

// Benchmark
// HiddenStates 2^1 - 2^13
// Observations 10^2 - 10^4


float gen_random_float()
{
    return (*gen)();
}

double gen_random_double()
{
    return (*gend)();
}

FLOAT_TYPE gen_random_real()
{
    return (FLOAT_TYPE)((*gend)());
}

FLOAT_TYPE log2(FLOAT_TYPE n)  
{  
    //log(n)/log(2) is log2.
    return (FLOAT_TYPE)log((FLOAT_TYPE)n) / (FLOAT_TYPE)log(2.0);
}

void initSmallHMM()
	{
		//std::cout<<sizeof(vfield_t)<<std::endl;

		ObservableStates = 64;

		// V[hiddenState * t]
		V = (vfield_t*) malloc(nObservations * HiddenStates * sizeof(vfield_t));
		//Vc = (vfield_t*) malloc(nObservations * HiddenStates * sizeof(vfield_t));
		// Path[t]
		Path = (cl_int*) malloc(nObservations * sizeof(cl_int));
		//PathC = (cl_int*) malloc(nObservations * sizeof(cl_int));
		

		// Observations[t]
		Observations = (cl_int *) malloc(nObservations * sizeof(cl_int));
		for (int i = 0; i < nObservations; ++i) {
			Observations[i] = rand() % ObservableStates;
		}
		/*
		Observations[0] = 0;
		Observations[1] = 1;
		Observations[2] = 2;
		Observations[3] = 0;
		Observations[4] = 0;*/

		// StartProbability[hiddenState]
		StartProbability = (FLOAT_TYPE*) malloc(HiddenStates * sizeof(FLOAT_TYPE));
		FLOAT_TYPE sum = (FLOAT_TYPE)0.0;
		for (int i = 0; i < HiddenStates; ++i) {
			StartProbability[i] = gen_random_real();
			sum += StartProbability[i];
		}
		for (int i = 0; i < HiddenStates; ++i) {
			StartProbability[i] = log2(StartProbability[i] / sum);
		}
		/*
		StartProbability[0] = log2(0.6);
		StartProbability[1] = log2(0.4);
		*/

		// TransitionMatrix[hiddenState][hiddenState]
		TransitionMatrix = (FLOAT_TYPE*) malloc(HiddenStates * HiddenStates * sizeof(FLOAT_TYPE));
		for (int i = 0; i < HiddenStates; ++i) {
			sum = (FLOAT_TYPE)0.0;
			for (int j = 0; j < HiddenStates; ++j) {
				TransitionMatrix[(i*HiddenStates)+j] = gen_random_real();
				sum += TransitionMatrix[(i*HiddenStates)+j];
			}
			for (int j = 0; j < HiddenStates; ++j) {
				TransitionMatrix[(i*HiddenStates)+j] = log2(TransitionMatrix[(i*HiddenStates)+j] / sum);
			}
		}

		/*
		TransitionMatrix[0 + 0] = log2(0.7);
		TransitionMatrix[0 + 1] = log2(0.3);

		TransitionMatrix[(1 * HiddenStates) + 0] = log2(0.4);
		TransitionMatrix[(1 * HiddenStates) + 1] = log2(0.6);
		*/

		// EmissionProbability[hiddenState][observableState]
		EmissionProbability = (FLOAT_TYPE*) malloc(HiddenStates * ObservableStates * sizeof(FLOAT_TYPE));
		for (int i = 0; i < HiddenStates; ++i) {
			sum = (FLOAT_TYPE)0.0;
			for (int j = 0; j < ObservableStates; ++j) {
				EmissionProbability[(i*ObservableStates)+j] = gen_random_real();
				sum += EmissionProbability[(i*ObservableStates)+j];
			}
			for (int j = 0; j < ObservableStates; ++j) {
				EmissionProbability[(i*ObservableStates)+j] = log2(EmissionProbability[(i*ObservableStates)+j] / sum);
			}
		}
		/*
		EmissionProbability[0 + 0] = log2(0.5);
		EmissionProbability[0 + 1] = log2(0.4);
		EmissionProbability[0 + 2] = log2(0.1);

		EmissionProbability[(1*ObservableStates) + 0] = log2(0.1);
		EmissionProbability[(1*ObservableStates) + 1] = log2(0.3);
		EmissionProbability[(1*ObservableStates) + 2] = log2(0.6);
		*/

		return;
	}

void buildFirstColumn()
{
	FLOAT_TYPE probability = 0.0;
	for(int i = 0; i < HiddenStates; i++)
	{
		probability = StartProbability[i] + EmissionProbability[(i*ObservableStates)+Observations[0]];
		V[i + 0].probability = probability;
		V[i + 0].state = -1;
	}
	return;
}

void runCPU()
{
	FLOAT_TYPE probability;
	// Initialization
	for(int i = 0; i < HiddenStates; i++)
	{
		probability = StartProbability[i] + EmissionProbability[(i*ObservableStates) + Observations[0]];
		Vc[i + 0].probability = probability;
		Vc[i + 0].state = -1;
	}

	FLOAT_TYPE emitp;
	// smallest possible double
	FLOAT_TYPE maxProbability = (FLOAT_TYPE)-REAL_MAX;
	cl_int maxState = -1;

	for(int t = 1; t < nObservations; t++)
	{
		for(int j = 0; j < HiddenStates; j++)
		{
			probability = (FLOAT_TYPE)REAL_MAX;
			maxProbability = (FLOAT_TYPE)-REAL_MAX;
			maxState = -1;
			emitp = EmissionProbability[(j*ObservableStates) + Observations[t]];

			for(int k = 0; k < HiddenStates; k++)
			{
				probability = Vc[k + ((t-1)*HiddenStates)].probability + TransitionMatrix[(k*HiddenStates) + j] + emitp;
				if(probability > maxProbability)
				{
					maxProbability = probability;
					maxState = k;
				}
			}
			Vc[j + (t*HiddenStates)].probability = maxProbability;
			Vc[j + (t*HiddenStates)].state = maxState;
		}
	}

	// find path by backtracking
	int lastRow = nObservations-1;
	maxProbability = (FLOAT_TYPE)-REAL_MAX;
	maxState = -1;
	for(int i = 0; i < HiddenStates; i++)
			{
				probability = Vc[i + (lastRow*HiddenStates)].probability;
				if(probability > maxProbability)
				{
					maxProbability = probability;
					maxState = (cl_int)i;
				}
			}
	PathProbabilityC = pow(2, maxProbability);
	PathC[lastRow] = maxState;
	int t = nObservations-2;
	while(t >= 0)
		{
			PathC[t] = Vc[PathC[t+1] + ((t+1)*HiddenStates)].state;
			t--;
		}
}

void FindGPUPath()
{
	FLOAT_TYPE probability;
	int lastRow = nObservations-1;
	FLOAT_TYPE maxProbability = -REAL_MAX;
	cl_int maxState = -1;
	for(int i = 0; i < HiddenStates; i++)
			{
				probability = V[i + (lastRow*HiddenStates)].probability;
				if(probability > maxProbability)
				{
					maxProbability = probability;
					maxState = i;
				}
			}
	PathProbability = pow(2, maxProbability);
	Path[lastRow] = maxState;
	int t = nObservations-2;
	while(t >= 0)
		{
			Path[t] = V[Path[t+1] + ((t+1)*HiddenStates)].state;
			t--;
		}
}

void cleanup()
{
	free(Observations);
	free(TransitionMatrix);
	free(EmissionProbability);
	free(V);
	//free(Vc);
	free(Path);
	//free(PathC);
}


int main(void)
{
	srand(GetTickCount());

	rng.seed(GetTickCount());
	u = boost::uniform_real<float>(0.0f, 10.0f);
	gen = new boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >(rng, u);

	rngd.seed(GetTickCount());
	ud = boost::uniform_real<double>(0.0, 10.0);
	gend = new boost::variate_generator<boost::mt19937&, boost::uniform_real<double> >(rngd, ud);

	ObservableStates = 0;
	HiddenStates = 0;
	nObservations = 0;
	Observations = NULL;
	TransitionMatrix = NULL;
	EmissionProbability = NULL;
	V = NULL;
	Vc = NULL;
	Path = NULL;
	PathC = NULL;
	PathProbability = 0.0;
	PathProbabilityC = 0.0;

	cl_int rc = 0;
	std::vector<cl::Platform> platformList;
	cl::Platform Platform;
	std::vector<cl::Device> Devices;
	cl::Device Device;
	std::vector<cl::Kernel> Kernels;
	cl_context_properties properties[3];
	cl::Context Context;
	cl::CommandQueue CommandQueue;
	cl::Program::Sources source;
	cl::Program Program;
	cl::Kernel Kernel;
	std::ifstream tt;
	std::string code;
	std::string info("");
	cl_int t;

	cl::Buffer TransitionMatrixBuffer;
	cl::Buffer EmissionProbBuffer;
	cl::Buffer VBuffer;

	//timer
	clock_t gpuStart, gpuEnd, prepareStart, prepareEnd, cpuStart, cpuEnd, benchmarkStart;
	benchmarkStart = clock();
	for(int i = 1; i <= 13; i++){
		HiddenStates = (cl_int)1 << i;
		for(int j = 1; j <= 4; j++){
			nObservations = (cl_int)pow(10.0f, j);
			std::cout << 
				"hs: " << HiddenStates <<
				" nObs: " << nObservations << std::endl;
			
			initSmallHMM();

			gpuStart = clock();
			buildFirstColumn();
			prepareStart = clock();
			cl::Platform::get(&platformList);
			Platform = platformList[1];
			//Platform.getInfo(CL_PLATFORM_NAME, &info);
			//std::cout << "Platform name: " << info << std::endl;
			properties[0] = CL_CONTEXT_PLATFORM;
			properties[1] = (cl_context_properties)(Platform)();
			properties[2] = 0;
			Context = cl::Context(CL_DEVICE_TYPE_ALL, properties);
			rc = Platform.getDevices(CL_DEVICE_TYPE_ALL, &Devices);
			Device = Devices[0];
			//Device.getInfo(CL_DEVICE_NAME, &info);
			//std::cout << "Device Name: " << info << std::endl;
			//Device.getInfo(CL_DEVICE_ADDRESS_BITS, &info);
			//std::cout << "address bits: " << (int)info[0] << std::endl;
			//std::vector<size_t> v;
			//Device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &v);
			//std::cout << "Work-item sizes: " << v[0] << v[1] << v[2] << std::endl;
			CommandQueue = cl::CommandQueue(Context, Device, 0);

			tt = std::ifstream("hmm.cl");
			code = std::string((std::istreambuf_iterator<char>(tt)), std::istreambuf_iterator<char>());
			source = cl::Program::Sources(1, std::make_pair(code.c_str(), code.size()));
			Program = cl::Program(Context, source);//, true, &rc);
			Program.build(Devices);

			Program.getBuildInfo(Device, CL_PROGRAM_BUILD_LOG, &info);
			//if(info.size() > 0)
				//std::cout << "Build log: " << info << std::endl;
		
			rc = Program.createKernels(&Kernels);
			Kernel = Kernels[0];

			prepareEnd = clock();
	
			TransitionMatrixBuffer = cl::Buffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HiddenStates * HiddenStates * sizeof(FLOAT_TYPE), (void*)TransitionMatrix, &rc);
			EmissionProbBuffer = cl::Buffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HiddenStates * ObservableStates * sizeof(FLOAT_TYPE), (void*)EmissionProbability, &rc);
			VBuffer = cl::Buffer(Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, HiddenStates * nObservations * sizeof(vfield_t), (void*)V, &rc);

			CommandQueue.enqueueWriteBuffer(EmissionProbBuffer, CL_FALSE, 0, HiddenStates * ObservableStates * sizeof(FLOAT_TYPE), (void*)EmissionProbability);
			CommandQueue.enqueueWriteBuffer(TransitionMatrixBuffer, CL_FALSE, 0, HiddenStates * HiddenStates * sizeof(FLOAT_TYPE), (void*)TransitionMatrix);
			CommandQueue.enqueueWriteBuffer(VBuffer, CL_FALSE, 0, HiddenStates * sizeof(vfield_t), (void*)V);

			for(t = 1; t<nObservations; t++)
			{
				Kernel.setArg(0, sizeof(cl_int), &t);
				Kernel.setArg(1, sizeof(cl_int), &Observations[t]);
				Kernel.setArg(2, sizeof(cl_int), &HiddenStates);
				Kernel.setArg(3, sizeof(cl_int), &ObservableStates);
				Kernel.setArg(4, EmissionProbBuffer);
				Kernel.setArg(5, TransitionMatrixBuffer);
				Kernel.setArg(6, VBuffer);
				rc = CommandQueue.enqueueNDRangeKernel(Kernel, cl::NullRange, cl::NDRange(HiddenStates), cl::NullRange);
				//rc = CommandQueue.enqueueNDRangeKernel(Kernel, cl::NullRange, cl::NDRange(HiddenStates), cl::NDRange(1));
			}

			CommandQueue.enqueueReadBuffer(VBuffer, CL_TRUE, 0, HiddenStates * nObservations * sizeof(vfield_t), V);
			FindGPUPath();
			gpuEnd = clock();

			std::cout<<"GPU time:"<< (gpuEnd-gpuStart) <<std::endl;

			//cpuStart = clock();
			//runCPU();
			//cpuEnd = clock();

			//std::cout<<"CPU time:"<< (cpuEnd-cpuStart) <<std::endl;
			//if (memcmp(Path, PathC, nObservations * sizeof(cl_int)) != 0)
				//printf("Error: Different Paths\n");

			std::cout << "GPU prepare time: " << (prepareEnd-prepareStart) << std::endl;
			cleanup();
		}
	}
	std::cout << "Overall time needed: " << (clock()-benchmarkStart) << std::endl;
	std::cin.get();
	return 0;
}
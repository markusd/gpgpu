#include <stdlib.h>

#include <vector>

#include <iostream>

#include "hmm.hpp"
#include <math.h>


HMM::HMM(bool useDefault)

{

	ObservableStates = 0;

	HiddenStates = 0;

	nObservations = 0;

	Observations = NULL;

	TransitionMatrix = NULL;

	EmissionProbability = NULL;

	V = NULL;

	Path = NULL;

	PathProbability = 0.0;

	if(useDefault)

	{

		initSmallHMM();

		buildFirstColumn();

	}

	cl_int rc = 0;

	std::string info("");

	std::vector<cl::Platform> platformList;

	cl::Platform::get(&platformList);

	cl::Platform Platform = platformList[0];

	Platform.getInfo(CL_PLATFORM_NAME, &info);

	std::cout << "Platform name: " << info << std::endl;

	cl_context_properties properties[] = {

			CL_CONTEXT_PLATFORM,

			(cl_context_properties)(platformList[0])(),

			0

	};

	cl::Context Context(CL_DEVICE_TYPE_GPU, properties);

	std::vector<cl::Device> Devices;

	rc = Platform.getDevices(CL_DEVICE_TYPE_GPU, &Devices);

	cl::Device Device(Devices[0]);

	Device.getInfo(CL_DEVICE_NAME, &info);

	std::cout << "Device Name: " << info << std::endl;

	cl::CommandQueue CommandQueue(Context, Device, 0);

	cl::Buffer TransitionMatrixBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HiddenStates * HiddenStates * sizeof(FLOAT_TYPE), (void*)TransitionMatrix, &rc);

	cl::Buffer EmissionProbBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HiddenStates * ObservableStates * sizeof(FLOAT_TYPE), (void*)EmissionProbability, &rc);

	cl::Buffer VBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, HiddenStates * nObservations * sizeof(vfield_t), (void*)V, &rc);

	SourceCode = "typedef struct vfield {\
	float probability;\
	int state;\
	} vfield_t;\
	\
__kernel void viterbiStep(__read_only int t,\
				__read_only int obs,\
				__read_only uint HiddenStates,\
				__read_only uint ObservableStates,\
                __constant __read_only float* EmissionProbability,\
				__constant __read_only float* TransitionMatrix,\
				__global vfield_t* V\
				   )\
		{\
            uint index = get_global_id(0);\
			\
			float maxProbability = -INFINITY;\
			float probability = 0.0f;\
			int maxState = -1;\
			float emitp = EmissionProbability[(index*ObservableStates) + obs];\
			\
			for(int i = 0; i < HiddenStates; i++)\
			{\
				probability = V[i + ((t-1)*HiddenStates)].probability + TransitionMatrix[(i*HiddenStates) + index] + emitp;\
				if(probability > maxProbability)\
				{\
					maxProbability = probability;\
					maxState = i;\
				}\
			}\
			\
			V[index + (t*HiddenStates)].probability = maxProbability;\
			V[index + (t*HiddenStates)].state = maxState;\
        }"; // TODO load viterbi.cl
	cl::Program::Sources source(1, std::make_pair(SourceCode.c_str(), SourceCode.size()));
	cl::Program Program(Context, source);//, true, &rc);

		Program.build(Devices);

		std::string info_("");
		Program.getBuildInfo(Devices[0], CL_PROGRAM_BUILD_LOG, &info_);
		if (info_.size() > 0)
			std::cout << "Build log: " << info_ << std::endl;

	std::vector<cl::Kernel> Kernels;

	rc = Program.createKernels(&Kernels);

	cl::Kernel Kernel = Kernels[0];

	if(useDefault)

	{

		unsigned int t;

		for(t = 1; t<nObservations; t++)

		{

			Kernel.setArg(0, sizeof(unsigned int), &t);

			Kernel.setArg(1, sizeof(int), &Observations[t]);

			Kernel.setArg(2, sizeof(unsigned int), &HiddenStates);

			Kernel.setArg(3, sizeof(unsigned int), &ObservableStates);

			Kernel.setArg(4, TransitionMatrixBuffer);

			Kernel.setArg(5, EmissionProbBuffer);

			Kernel.setArg(6, VBuffer);

			rc = CommandQueue.enqueueNDRangeKernel(Kernel, cl::NullRange, 16, cl::NullRange);

		}

		CommandQueue.enqueueReadBuffer(VBuffer, CL_TRUE, 0, HiddenStates * nObservations * sizeof(vfield_t), V);

		return;

	}

}





HMM::~HMM(void)

{

	free(Observations);

	free(TransitionMatrix);

	free(EmissionProbability);

	free(V);

	free(Path);

}





void HMM::initSmallHMM()

	{

		ObservableStates = 3;

		HiddenStates = 2;

		nObservations = 3;

		

		// Observations[t]

		Observations = (unsigned int *) malloc(nObservations * sizeof(unsigned int));

		Observations[0] = 0;

		Observations[1] = 1;

		Observations[2] = 2;



		// StartProbability[hiddenState]

		StartProbability = (FLOAT_TYPE*) malloc(HiddenStates * sizeof(FLOAT_TYPE));

		StartProbability[0] = log2(0.6);

		StartProbability[1] = log2(0.4);



		// TransitionMatrix[hiddenState][hiddenState]

		TransitionMatrix = (FLOAT_TYPE*) malloc(HiddenStates * HiddenStates * sizeof(FLOAT_TYPE));



		TransitionMatrix[0 + 0] = log2(0.7);

		TransitionMatrix[0 + 1] = log2(0.3);



		TransitionMatrix[(1 * HiddenStates) + 0] = log2(0.4);

		TransitionMatrix[(1 * HiddenStates) + 1] = log2(0.6);



		// EmissionProbability[hiddenState][observableState]

		EmissionProbability = (FLOAT_TYPE*) malloc(HiddenStates * ObservableStates * sizeof(FLOAT_TYPE*));

		

		EmissionProbability[0 + 0] = log2(0.5);

		EmissionProbability[0 + 1] = log2(0.4);

		EmissionProbability[0 + 2] = log2(0.1);



		EmissionProbability[(1*ObservableStates) + 0] = log2(0.1);

		EmissionProbability[(1*ObservableStates) + 1] = log2(0.3);

		EmissionProbability[(1*ObservableStates) + 2] = log2(0.6);



		// V[hiddenState * t]

		V = (vfield_t*) malloc(nObservations * HiddenStates * sizeof(vfield_t));

		Path = (unsigned int*) malloc(nObservations * sizeof(unsigned int));



		return;

	}



void HMM::buildFirstColumn()

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



FLOAT_TYPE HMM::log2(FLOAT_TYPE n)  

{  

    //log(n)/log(2) is log2.

    return log((FLOAT_TYPE)n) / log(2.0);

}
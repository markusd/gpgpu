#ifndef __HMM_HPP

	#define __HMM_HPP



#include <CL/cl.hpp>

#define FLOAT_TYPE cl_float



typedef struct vfield {

	FLOAT_TYPE probability;

	cl_int state;

	} vfield_t;



class HMM

{

private:

	// HMM

	unsigned int ObservableStates;

	unsigned int HiddenStates;

	unsigned int nObservations;

	unsigned int* Observations;		// vector

	FLOAT_TYPE* StartProbability;	// vector

	FLOAT_TYPE* TransitionMatrix;	// matrix[i][j] i-->j matrix[(i*HiddenStates) + j] 

	FLOAT_TYPE* EmissionProbability;// matrix[i][j] i-->j matrix[(i*ObservableStates) + j]

	// temporary

	vfield_t* V; // V[i][t] = V[i + (t*HiddenStates)]

	// HMM result

	unsigned int* Path; // vector

	FLOAT_TYPE PathProbability;

	// OpenCL stuff

	cl_int rc;

	std::string SourceCode;

	// methods

	void ViterbiRun();

	void ViterbiStep(unsigned int n);

	// helper

	void initSmallHMM();

	void buildFirstColumn();

	FLOAT_TYPE log2(FLOAT_TYPE n);

public:

	HMM(bool useDefault = false);

	~HMM(void);

	unsigned int* getPath(){return Path;};

	FLOAT_TYPE getPathProbability(){return PathProbability;};

};



#endif /* __HMM_HPP */
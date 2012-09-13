#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define FLOAT_TYPE double

typedef struct vfield {
	FLOAT_TYPE probability;
	int state;
	} vfield_t;
	
__kernel void viterbiStep(int t,
				int obs,
				int HiddenStates,
				int ObservableStates,
                __global FLOAT_TYPE* EmissionProbability,
				__global FLOAT_TYPE* TransitionMatrix,
				__global vfield_t* V
				   )
{
	int index = get_global_id(0);
	//int globalId = get_global_id(0);
	//int localSize = get_local_size(0);
	//printf("%d\n", get_local_size(0));
	if (index >= HiddenStates) {
		return;
	}

	//for(int index = globalId; index < HiddenStates; index += localSize){
		FLOAT_TYPE probability = INFINITY;
		FLOAT_TYPE maxProbability = -INFINITY;
		int maxState = -1;
		FLOAT_TYPE emitp = EmissionProbability[(index*ObservableStates) + obs];
		for(int i = 0; i < HiddenStates; i++)
		{
			probability = V[i + ((t-1)*HiddenStates)].probability + TransitionMatrix[(i*HiddenStates) + index] + emitp;
			if(probability > maxProbability)
			{
				maxProbability = probability;
				maxState = i;
			}
		}
		V[index + (t*HiddenStates)].probability = maxProbability;
		V[index + (t*HiddenStates)].state = maxState;
	//}
}
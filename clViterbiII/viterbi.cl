typedef struct vfield {
	float probability;
	int state;
	} vfield_t;

__kernel void viterbiStep(__read_only int t,
				__read_only int obs,
				__read_only uint HiddenStates,
				__read_only uint ObservableStates,
                __constant __read_only float* EmissionProbability,
				__constant __read_only float* TransitionMatrix,
				__global vfield_t* V
				   )
		{
            uint index = get_global_id(0);
			
			float probability = INFINITY;
			float maxProbability = -INFINITY;
			int maxState = -1;
			float emitp = EmissionProbability[(index*ObservableStates) + obs];
			
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
        }

float distsqr(__global float* a, __global float* b)
{
	float result = 0.0f;

	for (int i = 0; i < DIM; ++i) {
		result += (a[i] - b[i]) * (a[i] - b[i]);
	}

	return result;
}

void copy(__local float* dst, __global float* src)
{
	for (int i = 0; i < DIM; ++i) {
		dst[i] = src[i];
	}
}

__kernel void compute_mean(__global float* input, __global float* mean)
{
	int id = get_global_id(0);

	float comp = 0.0f;
	for (int i = 0; i < N; ++i)
		comp += input[i*DIM+id];

	mean[id] = comp / (float)N;
}

__kernel void cluster_assignment(__global float* input, __global float* centroids, __global int* mapping)
{
	//__local float cache[8192];
	int id = get_global_id(0);
	//int lid = get_local_id(0);

	//if (lid < K) {
	//	copy(&cache[lid*DIM], &centroids[lid*DIM]);
	//}

	// nearest centroid id
	int cid = 0;
	float cdist = 999999.0f;

	// for each centroid, calculate distance
#pragma unroll
	for (int i = 0; i < K; ++i) {
		float dist = distsqr(&input[id*DIM], &centroids[i*DIM]);
		if (dist < cdist) {
			cid = i;
			cdist = dist;
		}
	}

	mapping[id] = cid;
}


/**
 * Called DIM times in 1D.
 */
__kernel void cluster_reposition(__global float* input, __global int* mapping, __global float* new_centroids)
{
	//__local int mapping_cache[N];
	int id = get_global_id(0);
	int lid = get_local_id(0);

//#pragma unroll
//	for (int i = 0; i < N / 16; ++i)
//		mapping_cache[i*16+lid] = mapping[i*16+lid];

	//barrier(CLK_LOCAL_MEM_FENCE);

	float comp[K];
	float count[K];

#pragma unroll
	for (int i = 0; i < K; ++i) {
		comp[i] = 0.0f;
		count[i] = 0.0f;
	}

	for (int i = 0; i < N; ++i) {
		comp[mapping[i]] += input[i*DIM+id];
		count[mapping[i]] += 1.0f;
	}

#pragma unroll
	for (int i = 0; i < K; ++i)
		new_centroids[i*DIM+id] = comp[i] / max(count[i], 1.0f);
}
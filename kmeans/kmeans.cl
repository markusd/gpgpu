//#define USE_LOCAL
//#define USE_LOCAL_MAPPING

#ifdef USE_LOCAL
#define CENTROID_MEM_TYPE __local
#else
#define CENTROID_MEM_TYPE __global
#endif

#define DIST_MEASURE distsqr

float distsqr(__global float* a, CENTROID_MEM_TYPE float* b)
{
	float result = 0.0f;
//#pragma unroll
	for (int i = 0; i < DIM; ++i) {
		result += (a[i] - b[i]) * (a[i] - b[i]);
	}

	return result;
}

float distsqr__(float* a, __global float* b)
{
	float result = 0.0f;

	for (int i = 0; i < DIM; ++i) {
		result += (a[i] - b[i]) * (a[i] - b[i]);
	}

	return result;
}

float distcos(__global float* a, CENTROID_MEM_TYPE float* b)
{
float result = 0.0f, lena = 0.0f, lenb = 0.0f;

	for (int i = 0; i < DIM; ++i) {
		result += a[i] * b[i];
		lena += a[i] * a[i];
		lenb += b[i] * b[i];
	}

	return 1.0f - result / (sqrt(lena) * sqrt(lenb));
}

float disttan(__global float* a, CENTROID_MEM_TYPE float* b)
{
	float result = 0.0f;
	float lena = 0.0f;
	float lenb = 0.0f;

	for (int i = 0; i < DIM; ++i) {
		result += a[i] * b[i];
		lena += a[i] * a[i];
		lenb += b[i] * b[i];
	}

	return 1.0f - result / (lena + lenb - result);
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

__kernel void compute_cost(__global float* input, __global float* centroids, __global int* mapping)
{

}

__kernel void compare_mapping(__global int* mapping, __global int* old_mapping, __global int* changed)
{
	// when to set changed to 0
	int id = get_global_id(0);
	if (mapping[id] != old_mapping[id])
		changed[0] = 1;
}

__kernel void cluster_assignment(__global float* input, __global float* centroids, __global int* mapping)
{
#ifdef USE_LOCAL
	__local float cache[K*DIM];

	int lid = get_local_id(0);

	//if (lid < K) {
	//	copy(&cache[lid*DIM], &centroids[lid*DIM]);
	//}

	//for (int i = 0; i < K; i += AM_LWS)
	//	if (i + lid < K) {
	//	copy(&cache[(i+lid)*DIM], &centroids[(i+lid)*DIM]);
	//}

	for (int i = lid; i < K*DIM; i += AM_LWS)
		cache[i] = centroids[i];

	barrier(CLK_LOCAL_MEM_FENCE);
#endif

	int id = get_global_id(0);

	//if (id == 0)
	//	changed[0] = 0;


	// nearest centroid id
	int cid = 0;
	float cdist = 999999.0f;

	// for each centroid, calculate distance
#pragma unroll
	for (int i = 0; i < K; ++i) {
#ifdef USE_LOCAL
		float dist = DIST_MEASURE(&input[id*DIM], &cache[i*DIM]);//&centroids[i*DIM]);
#else
		float dist = DIST_MEASURE(&input[id*DIM], &centroids[i*DIM]);
#endif
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
	float comp[K*2];
	//float count[K];

#ifdef USE_LOCAL_MAPPING
	__local int mapping_cache[N];
	int lid = get_local_id(0);

	for (int i = 0; i < N / RP_LWS; ++i)
		mapping_cache[i*RP_LWS+lid] = mapping[i*RP_LWS+lid];

	//for (int i = lid; i < N; i += RP_LWS)
	//	mapping_cache[i] = mapping[i];

	barrier(CLK_LOCAL_MEM_FENCE);
#endif

	int id = get_global_id(0);



#pragma unroll
	for (int i = 0; i < K; ++i) {
		comp[i] = 0.0f;
		comp[i+K] = 0.0f;
		//count[i] = 0.0f;
	}

	for (int i = 0; i < N; ++i) {
#ifdef USE_LOCAL_MAPPING
		comp[mapping_cache[i]] += input[i*DIM+id];
		comp[mapping_cache[i]+K] += 1.0f;
		//count[mapping_cache[i]] += 1.0f;
#else
		comp[mapping[i]] += input[i*DIM+id];
		comp[mapping[i]+K] += 1.0f;
		//count[mapping[i]] += 1.0f;
#endif
	}

#pragma unroll
	for (int i = 0; i < K; ++i)
		new_centroids[i*DIM+id] = comp[i] / comp[i+K];
}


__kernel void cluster_reposition_k(__global float* input, __global int* mapping, __global float* new_centroids)
{
	int id = get_global_id(0);
	//int lid = get_local_id(0);
	float count = 0.0f;

	float ctrd[DIM];
	//__local float ctrd[DIM*RP_LWS];

#pragma unroll
	for (int i = 0; i < DIM; ++i)
		//ctrd[lid*i+DIM] = 0.0f;
		//new_centroids[id*DIM+i] = 0.0f;
		ctrd[i] = 0.0f;

	for (int i = 0; i < N; ++i) {
		if (mapping[i] == id) {
			count += 1.0f;
			for (int j = 0; j < DIM; ++j) {
				//ctrd[lid*j+DIM] += input[i*DIM+j];
				ctrd[j] += input[i*DIM+j];
				//new_centroids[id*DIM+j] += input[i*DIM+j];
			}
		}
	}

#pragma unroll
	for (int i = 0; i < DIM; ++i)
		//new_centroids[id*DIM+i] = ctrd[lid*i+DIM] / count;
		new_centroids[id*DIM+i] = ctrd[i] / count;
		//new_centroids[id*DIM+i] /= count;
	
}

__kernel void c_cluster_reposition(__global float* input, __global int* mapping, __global float* new_centroids, __global int* converged)
{
	int id = get_global_id(0);
	//int lid = get_local_id(0);
	float count = 0.0f;

	float ctrd[DIM];
	//__local float ctrd[DIM*RP_LWS];

#pragma unroll
	for (int i = 0; i < DIM; ++i)
		//ctrd[lid*i+DIM] = 0.0f;
		//new_centroids[id*DIM+i] = 0.0f;
		ctrd[i] = 0.0f;

	for (int i = 0; i < N; ++i) {
		if (mapping[i] == id) {
			count += 1.0f;
			for (int j = 0; j < DIM; ++j) {
				//ctrd[lid*j+DIM] += input[i*DIM+j];
				ctrd[j] += input[i*DIM+j];
				//new_centroids[id*DIM+j] += input[i*DIM+j];
			}
		}
	}

	for (int i = 0; i < DIM; ++i)
		ctrd[i] /= count;

	if (distsqr__(ctrd, &new_centroids[id*DIM]) > 0.01f)
		converged[0] = 0;

#pragma unroll
	for (int i = 0; i < DIM; ++i)
		//new_centroids[id*DIM+i] = ctrd[lid*i+DIM] / count;
		new_centroids[id*DIM+i] = ctrd[i];
		//new_centroids[id*DIM+i] /= count;
	
}

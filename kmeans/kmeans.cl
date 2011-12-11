float distsqr(__global float* a, __global float* b, int dim)
{
	float result = 0.0f;
	for (int i = 0; i < dim; ++i) {
		result += (a[i] - b[i]) * (a[i] - b[i]);
	}

	return result;
}

void copy(__local float* dst, __global float* src, int dim)
{
	for (int i = 0; i < dim; ++i) {
		dst[i] = src[i];
	}
}

__kernel void compute_mean(__global float* input, int dim, int n, __global float* mean)
{
	int id = get_global_id(0);

	float comp = 0.0f;
	for (int i = 0; i < n; ++i)
		comp += input[i*dim+id];

	mean[id] = comp / (float)n;
}

__kernel void cluster_assignment(__global float* input, int dim, int n, __global float* centroids, int k, __global int* mapping)
{
	//__local float cache[8192];
	int id = get_global_id(0);
	//int lid = get_local_id(0);

	//if (lid < k) {
	//	copy(&cache[lid*dim], &centroids[lid*dim], dim);
	//}

	// nearest centroid id
	int cid = 0;
	float cdist = 999999.0f;

	// for each centroid, calculate distance
	for (int i = 0; i < k; ++i) {
		float dist = distsqr(&input[id*dim], &centroids[i*dim], dim);
		if (dist < cdist) {
			cid = i;
			cdist = dist;
		}
	}

	mapping[id] = cid;
}

__kernel void cluster_reposition(__global float* input, int dim, int n, __global float* centroids, int k, __global int* mapping)
{
}
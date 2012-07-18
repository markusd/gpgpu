#include "kmeans.cuh"
#include <kmeans_def.hpp>

#define DIST_MEASURE distsqr

__device__ float distsqr(const float* a, const float* b)
{
	float result = 0.0f;

	for (int i = 0; i < DIM; ++i) {
		result += (a[i] - b[i]) * (a[i] - b[i]);
	}

	return result;
}

__device__ float distcos(const float* a, const float* b)
{
	float result = 0.0f;
	float lena = 0.0f;
	float lenb = 0.0f;

	for (int i = 0; i < DIM; ++i) {
		result += a[i] * b[i];
		lena += a[i] * a[i];
		lenb += b[i] * b[i];
	}

	return 1.0f - result / (sqrt(lena) * sqrt(lenb));
}

__device__ float disttan(const float* a, const float* b)
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



__device__ void copy(float* dst, const float* src)
{
	for (int i = 0; i < DIM; ++i) {
		dst[i] = src[i];
	}
}

__global__ void cuda_assignment(const float* input, const float* centroids, int* mapping)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	// nearest centroid id
	int cid = 0;
	float cdist = 999999.0f;

	// for each centroid, calculate distance
#pragma unroll
	for (int i = 0; i < K; ++i) {

		float dist = DIST_MEASURE(&input[id*DIM], &centroids[i*DIM]);

		if (dist < cdist) {
			cid = i;
			cdist = dist;
		}
	}

	mapping[id] = cid;
}


void cuda_launch_assignment(const float* input, const float* centroids, int* mapping)
{
	dim3 block(8, 1, 1);
	dim3 grid(N / block.x, 1, 1);
	cuda_assignment<<<grid, block>>>(input, centroids, mapping);
}

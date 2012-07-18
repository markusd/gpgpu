#ifndef KMEANS_CUH_
#define KMEANS_CUH_

void cuda_launch_assignment(const float* input, const float* centroids, int* mapping);
void cuda_launch_reposition(const float* input, float* centroids, const int* mapping);

#endif
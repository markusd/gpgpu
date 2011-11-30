#ifndef WAVES_CUH_
#define WAVES_CUH_

void cuda_launch_waves(float* out, int width, float dt, int count, const float* pos);

#endif
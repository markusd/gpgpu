#include "waves.cuh"

__global__ void cuda_waves(float* out, int width, float dt, int count, const float* pos)
{
	dt *= 0.333333333f;

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	float tx = (float)x / (float)width;// + m_xoffs;
	float ty = (float)y / (float)width;// + m_yoffs;

	float amp = 0.0f;
	float waves = 2.5f / (float)count;

	for (int i = 0; i < count; ++i) {
		float dx = tx - pos[i*2];
		float dy = ty - pos[i*2+1];
		float dist = sqrt(dx*dx + dy*dy);
		amp += waves * sin(6.2831f * (dt - dist * 5.0f));
	}

	float r = 0.0f;
	float g = 0.0f;
	float b = 0.0f;

	amp = 2.0f * (amp < 0.0f ? -amp : amp);

	if (amp <= 1.0f) {
		r = 1.0f;
		g = amp;
	} else if (amp <= 2.0f) {
		r = (1.96f - amp);
		g = 1.0f;
	} else if (amp <= 3.0f) {
		g = 1.0f;
		b = (amp - 2.0f);
	} else if (amp <= 4.0f) {
		g = (4.0f - amp);
		b = 1.0f;
	} else {
		r = (amp - 4.01f);
		b = 1.0f;
	}

	out[(x + y * width) * 4] = 1.0f - r;
	out[(x + y * width) * 4 + 1] = 1.0f - g;
	out[(x + y * width) * 4 + 2] = 1.0f - b;
	out[(x + y * width) * 4 + 3] = 0.0f;
}



void cuda_launch_waves(float* out, int width, float dt, int count, const float* pos)
{
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, width / block.y, 1);
	cuda_waves<<<grid, block>>>(out, width, dt, count, pos);
}
#include "edges.cuh"

#define SQR(x) ((x)*(x))

__global__ void cuda_edges(PIXEL* in, int w, int h, PIXEL* out)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	float horz = in[x+y*w] - in[max(0, x-1)+y*w];
	float vert = in[x+y*w] - in[x+max(0, y-1)*w];
	out[x+y*w] = sqrt(SQR(horz) + SQR(vert));
}


void cuda_launch_edges(PIXEL* in, int w, int h, PIXEL* out)
{
	dim3 block(16, 16, 1);
	dim3 grid(w / block.x, h / block.y, 1);
	cuda_edges<<<grid, block>>>(in, w, h, out);
}


#ifdef USE_SHARED

#define SMEM(X, Y) cache[(X)+(Y)*ts]


__global__ void cuda_sobel(PIXEL* in, int w, int h, int ts, PIXEL* out)
{
	extern __shared__ PIXEL cache[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	SMEM(1+tx,1+ty) = in[x+y*w];

	if (threadIdx.x < 1) {	
		SMEM(tx, 1 + ty) = in[x-1+y*w]; // left
		SMEM(1 + bw + tx, 1 + ty) = in[x+bw+y*w]; // right
	}

	if (threadIdx.y < 1) {   
		SMEM(1 + tx, ty) = in[x+(y-1)*w]; // top	 
		SMEM(1 + tx, 1 + bh + ty) = in[x+(y+bh)*w]; // bottom
	}

	if ((threadIdx.x < 1) && (threadIdx.y < 1)) {
		SMEM(tx, ty) = in[x-1+(y-1)*w]; // tl
		SMEM(tx, 1 + bh + ty) = in[x-1+(y+bh)*w]; // bl
		SMEM(1 + bw + tx, ty) = in[x+bh+(y-1)*w]; // tr
		SMEM(1 + bw + tx, 1 + bh + ty) = in[x+bw+(y+bh)*w]; // br
	}

	__syncthreads();

	
	int il = tx-1+1;
	int ir = tx+1+1;
	int it = ty+1+1;
	int ib = ty-1+1;

	
	float tl = SMEM(il,it);
	float tr = SMEM(ir,it);
	float bl = SMEM(il,ib);
	float br = SMEM(ir,ib);

	float a1 = 
		2.0f*SMEM(2+tx,1+ty) + tr + br - 
		2.0f*SMEM(tx,1+ty) - tl - bl;

	float a2 =
		2.0f*SMEM(1+tx,2+ty) + tr + tl - 
		2.0f*SMEM(1+tx,ty) - br - bl;
		

	out[x+y*w] = sqrt(a1*a1 + a2*a2);

}

#else /* USE_SHARED */

__global__ void cuda_sobel(PIXEL* in, int w, int h, PIXEL* out)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	/*
	int il = max(0,x-1);
	int ir = min(w-1,x+1);
	int it = min(h-1,y+1)*w;
	int ib = max(0,y-1)*w;

	
	float tl = in[it+il];
	float tr = in[it+ir];
	float bl = in[ib+il];
	float br = in[ib+ir];

	float a1 = 
		2.0f*in[ir+y*w] + tr + br - 
		2.0f*in[il+y*w] - tl - bl;

	float a2 =
		2.0f*in[x+it] + tr + tl - 
		2.0f*in[x+ib] - br - bl;

*/


/*
	float tl = in[max(0,x-1)+min(h-1,y+1)*w];
	float tr = in[min(w-1,x+1)+min(h-1,y+1)*w];
	float bl = in[max(0,x-1)+max(0,y-1)*w];
	float br = in[min(w-1,x+1)+max(0,y-1)*w];

	float a1 = 
		2.0f*in[min(w-1,x+1)+y*w] + tr + br - 
		2.0f*in[max(0,x-1)+y*w] - tl - bl;

	float a2 =
		2.0f*in[x+min(h-1,y+1)*w] + tr + tl - 
		2.0f*in[x+max(0, y-1)*w] - br - bl;
		*/

/*
	float a1 = 
		2.0f*in[min(w-1,x+1)+y*w] + in[ir+it] + in[ir+ib] - 
		2.0f*in[max(0,x-1)+y*w] - in[il+it] - in[il+ib];

	float a2 =
		2.0f*in[x+min(h-1,y+1)*w] + in[ir+it] + in[il+it] - 
		2.0f*in[x+max(0, y-1)*w] - in[ir+ib] - in[il+ib];
*/
	
	float a1 = 
		2.0f*in[min(w-1,x+1)+y*w] + in[min(w-1,x+1)+min(h-1,y+1)*w] + in[min(w-1,x+1)+max(0,y-1)*w] - 
		2.0f*in[max(0,x-1)+y*w] - in[max(0,x-1)+min(h-1,y+1)*w] - in[max(0,x-1)+max(0,y-1)*w];

	float a2 =
		2.0f*in[x+min(h-1,y+1)*w] + in[min(w-1,x+1)+min(h-1,y+1)*w] + in[max(0,x-1)+min(h-1,y+1)*w] - 
		2.0f*in[x+max(0, y-1)*w] - in[min(w-1,x+1)+max(0, y-1)*w] - in[max(0,x-1)+max(0, y-1)*w];


	out[x+y*w] = sqrt(a1*a1 + a2*a2);
}

#endif /* USE_SHARED */

void cuda_launch_sobel(PIXEL* in, int w, int h, PIXEL* out)
{
	dim3 block(16, 16, 1);
	dim3 grid(w / block.x, h / block.y, 1);

#ifdef USE_SHARED
	cuda_sobel<<<grid, block, (block.x+2)*(block.y+2)*sizeof(PIXEL)>>>(in, w, h, block.x+2, out);
#else
	cuda_sobel<<<grid, block>>>(in, w, h, out);
#endif
}



__global__ void cuda_harris(PIXEL* in, int w, int h, PIXEL* out)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	float horz = in[x+y*w] - in[max(0, x-1)+y*w];
	float vert = in[x+y*w] - in[x+max(0, y-1)*w];
	out[x+y*w] = sqrt(SQR(horz) + SQR(vert));
}

void cuda_launch_harris(PIXEL* in, int w, int h, PIXEL* out)
{
	dim3 block(16, 16, 1);
	dim3 grid(w / block.x, h / block.y, 1);
	cuda_harris<<<grid, block>>>(in, w, h, out);
}
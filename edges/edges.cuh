#ifndef EDGES_CUH_
#define EDGES_CUH_

#include "edges.hpp"

void cuda_launch_grayscale(uchar4* in, int w, int h, PIXEL* out);
void cuda_launch_blur(PIXEL* in, int w, int h, PIXEL* out);

void cuda_launch_localmaxima(PIXEL* in, int w, int h, PIXEL* out);
void cuda_launch_localmaxima2(PIXEL* in, int w, int h, PIXEL* out);

void cuda_launch_edges(PIXEL* in, int w, int h, PIXEL* out);

void cuda_launch_sobel(PIXEL* in, int w, int h, PIXEL* out);

void cuda_launch_harris(PIXEL* in, int w, int h, PIXEL* out);

#endif
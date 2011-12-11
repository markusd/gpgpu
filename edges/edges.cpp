#include <iostream>
#include <opengl/stb_image.hpp>

#include <windows.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <edges.cuh>

#include <util/clock.hpp>


const float k = 0.04f;
const float eps = 1.1920929E-07f;
#define SQR(x) ((x)*(x))



int main(int argc, char** argv)
{
	//if (argc < 3) {
	//	std::cout <<
	//		"Usage: edges <in> <out>\n" <<
	//		std::endl;
	//}

	std::string in_file = "city.bmp"; //argv[1];
	std::string out_file = "city.harris.bmp"; //argv[2];



	int w, h, c;
	unsigned char* in_data = stbi_load(in_file.c_str(), &w, &h, &c, STBI_rgb_alpha);

	if (!in_data) {
		std::cout << "Error: Could not read input file" << std::endl;
		return 1;
	}

	// convert to grayscale image
	PIXEL* gray_scale = new PIXEL[w * h];
	for (int x = 0; x < w; ++x) {
		for (int y = 0; y < h; ++y) {
#ifdef USE_FLOAT
			gray_scale[x+y*w] = (0.3f * in_data[(x+y*h)*4+0] + 0.59f * in_data[(x+y*h)*4+1] + 0.11f * in_data[(x+y*h)*4+2]) / 255.0f;
#else
			gray_scale[x+y*w] = 0.3f * in_data[(x+y*h)*4+0] + 0.59f * in_data[(x+y*h)*4+1] + 0.11f * in_data[(x+y*h)*4+2];
#endif
		}
	}

	free(in_data);

	// edge data
	PIXEL* edge_detect = new PIXEL[w * h];

	util::Clock clock;
	clock.reset();
	
#ifdef USE_CUDA
	util::Clock up_clock;
	up_clock.reset();

	PIXEL* cuDevIn = NULL;
	cudaMalloc((void **)&cuDevIn, w * h * sizeof(PIXEL));
	cudaMemcpy(cuDevIn, gray_scale, w * h * sizeof(PIXEL), cudaMemcpyHostToDevice);

	PIXEL* cuDevOut = NULL;
	cudaMalloc((void **)&cuDevOut, w * h * sizeof(PIXEL));
	cudaThreadSynchronize();
	float up_time = up_clock.get();
	std::cout << up_time << std::endl;
	up_clock.reset();

	cuda_launch_sobel(cuDevIn, w, h, cuDevOut);
	cudaThreadSynchronize();
	up_time = up_clock.get();
	std::cout << up_time << std::endl;
	up_clock.reset();

	cudaMemcpy(edge_detect, cuDevOut, w * h * sizeof(PIXEL), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	up_time = up_clock.get();
	std::cout << up_time << std::endl;
#else


	// find edges
	for (int x = 0; x < w; ++x) {
		for (int y = 0; y < h; ++y) {
			//edge_detect[x+y*w] = sqrt(
			//	SQR(gray_scale[x+y*w] - gray_scale[max(0, x-1)+y*w]) +
			//	SQR(gray_scale[x+y*w] - gray_scale[x+max(0, y-1)*w]));

			
			//edge_detect[x+y*w] = det(A) - k * spur(A)^2;
			float a1 = 
				2.0f*gray_scale[min(w-1,x+1)+y*w] + gray_scale[min(w-1,x+1)+min(h-1,y+1)*w] + gray_scale[min(w-1,x+1)+max(0,y-1)*w] - 
				2.0f*gray_scale[max(0,x-1)+y*w] - gray_scale[max(0,x-1)+min(h-1,y+1)*w] - gray_scale[max(0,x-1)+max(0,y-1)*w];
			
			//float a2 = gray_scale[x+y*w] - gray_scale[x+max(0, y-1)*w];

			float a2 =
				2.0f*gray_scale[x+min(h-1,y+1)*w] + gray_scale[min(w-1,x+1)+min(h-1,y+1)*w] + gray_scale[max(0,x-1)+min(h-1,y+1)*w] - 
				2.0f*gray_scale[x+max(0, y-1)*w] - gray_scale[min(w-1,x+1)+max(0, y-1)*w] - gray_scale[max(0,x-1)+max(0, y-1)*w];

			edge_detect[x+y*w] = sqrt(a1*a1 + a2*a2);
			
			//float a3 = a1 * a2;
			//a1 *= a1;
			//a2 *= a2;

			//edge_detect[x+y*w] = (a1 * a2 - a3 * a3) - k * SQR(a1 + a2);

			//float e1 = 0.5f * (a1 + a2) + 0.5f * sqrt(4.0f * a3 * a3 + SQR(a1 - a2));
			//float e2 = 0.5f * (a1 + a2) - 0.5f * sqrt(4.0f * a3 * a3 + SQR(a1 - a2));

			//edge_detect[x+y*w] = e1*e2 - k*SQR(e1+e2);
			
		}
	}
#endif

	float time = clock.get();
	std::cout << time << std::endl;


	// copy to output buffer
	unsigned char* out_data = new unsigned char[w * h * 3];
	unsigned int out_size = w * h * 3;

	for (int x = 0; x < w; ++x) {
		for (int y = 0; y < h; ++y) {
#ifdef USE_FLOAT
			out_data[(x+y*w)*3+0] = out_data[(x+y*w)*3+1] = out_data[(x+y*w)*3+2] = (unsigned char)(edge_detect[x+y*w] * 255.0f);
#else
			out_data[(x+y*w)*3+0] = out_data[(x+y*w)*3+1] = out_data[(x+y*w)*3+2] = edge_detect[x+y*w];
#endif
		}
	}

	FILE *out_fdesc = fopen(out_file.c_str(), "wb");
	if (out_fdesc == NULL) {
		std::cout << "Error: Could not create output file" << std::endl;
		return 1;
	}

	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER infoHeader;

	fileHeader.bfType	  = 0x4d42;
	fileHeader.bfSize	  = 0;
	fileHeader.bfReserved1 = 0;
	fileHeader.bfReserved2 = 0;
	fileHeader.bfOffBits   = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	infoHeader.biSize		  = sizeof(infoHeader);
	infoHeader.biWidth		 = w;
	infoHeader.biHeight		= h;
	infoHeader.biPlanes		= 1;
	infoHeader.biBitCount	  = 24;
	infoHeader.biCompression   = BI_RGB;
	infoHeader.biSizeImage	 = 0;
	infoHeader.biXPelsPerMeter = 0;
	infoHeader.biYPelsPerMeter = 0;
	infoHeader.biClrUsed	   = 0;
	infoHeader.biClrImportant  = 0;

	fwrite((char*)&fileHeader, sizeof(fileHeader), 1, out_fdesc);
	fwrite((char*)&infoHeader, sizeof(infoHeader), 1, out_fdesc);

	// first row is saved last in a bitmap
	for (int y = 0; y < h; ++y)
		fwrite(&out_data[(h-1-y)*w*3], sizeof(unsigned char), w * 3, out_fdesc);
	fclose(out_fdesc);

	std::cin.get();
	return 0;
}
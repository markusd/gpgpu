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

	std::string in_file = "pepper.bmp"; //argv[1];
	std::string out_file = "pepper.test.bmp"; //argv[2];



	int w, h, c;
	unsigned char* in_data = stbi_load(in_file.c_str(), &w, &h, &c, STBI_rgb_alpha);

	if (!in_data) {
		std::cout << "Error: Could not read input file" << std::endl;
		return 1;
	}

	//cudaDeviceSynchronize();

#ifdef USE_CUDA
	util::Clock clock;
	clock.reset();

	uchar4* cuDevIn = NULL;
	cudaMalloc((void **)&cuDevIn, w * h * sizeof(uchar4));
	cudaMemcpy(cuDevIn, in_data, w * h * sizeof(uchar4), cudaMemcpyHostToDevice);

	PIXEL* cuDevBuf1 = NULL;
	cudaMalloc((void **)&cuDevBuf1, w * h * sizeof(PIXEL));

	cuda_launch_grayscale(cuDevIn, w, h, cuDevBuf1);

	PIXEL* cuDevBuf2 = NULL;
	cudaMalloc((void **)&cuDevBuf2, w * h * sizeof(PIXEL));

	cuda_launch_blur(cuDevBuf1, w, h, cuDevBuf2);

	cuda_launch_sobel(cuDevBuf2, w, h, cuDevBuf1);

	cuda_launch_localmaxima(cuDevBuf1, w, h, cuDevBuf2);


	PIXEL* result_data = new PIXEL[w * h];

	cudaMemcpy(result_data, cuDevBuf2, w * h * sizeof(PIXEL), cudaMemcpyDeviceToHost);

	float time = clock.get();
	std::cout << time << std::endl;

	cudaFree(cuDevBuf1);
	cudaFree(cuDevBuf2);
	cudaFree(cuDevIn);

#else /* USE_CUDA */

		// convert to grayscale image
	PIXEL* gray_scale = new PIXEL[w * h];

	for (int x = 0; x < w; ++x) {
		for (int y = 0; y < h; ++y) {
#ifdef USE_FLOAT
			gray_scale[x+y*w] = (0.3f * in_data[(x+y*w)*4+0] + 0.59f * in_data[(x+y*w)*4+1] + 0.11f * in_data[(x+y*w)*4+2]) / 255.0f;
#else
			gray_scale[x+y*w] = 0.333f * in_data[(x+y*w)*4+0] + 0.333f * in_data[(x+y*w)*4+1] + 0.333f * in_data[(x+y*w)*4+2];
#endif
		}
	}

	
	//free(in_data);

	// edge data
	PIXEL* result_data = new PIXEL[w * h];

	util::Clock clock;
	clock.reset();
	float time = 0.0f;

	
	// find edges
	for (int x = 0; x < w; ++x) {
		for (int y = 0; y < h; ++y) {
			result_data[x+y*w] = sqrt((double)
				SQR(gray_scale[x+y*w] - gray_scale[max(0, x-1)+y*w])  * 2.0 +
				SQR(gray_scale[x+y*w] - gray_scale[x+max(0, y-1)*w])  * 2.0) * 1.2;

			//result_data[x+y*w] =
			//	abs(gray_scale[x+y*w] - gray_scale[x+max(0, y-1)*w]) * 2.0f;

			
			//result_data[x+y*w] = det(A) - k * spur(A)^2;

			
			float a1 = 
				2.0f*gray_scale[min(w-1,x+1)+y*w] + gray_scale[min(w-1,x+1)+min(h-1,y+1)*w] + gray_scale[min(w-1,x+1)+max(0,y-1)*w] - 
				2.0f*gray_scale[max(0,x-1)+y*w] - gray_scale[max(0,x-1)+min(h-1,y+1)*w] - gray_scale[max(0,x-1)+max(0,y-1)*w];
			
			//float a2 = gray_scale[x+y*w] - gray_scale[x+max(0, y-1)*w];

			float a2 =
				2.0f*gray_scale[x+min(h-1,y+1)*w] + gray_scale[min(w-1,x+1)+min(h-1,y+1)*w] + gray_scale[max(0,x-1)+min(h-1,y+1)*w] - 
				2.0f*gray_scale[x+max(0, y-1)*w] - gray_scale[min(w-1,x+1)+max(0, y-1)*w] - gray_scale[max(0,x-1)+max(0, y-1)*w];

			//result_data[x+y*w] = sqrt(a1*a1 + a2*a2);
			
			
			//float a3 = a1 * a2;
			//a1 *= a1;
			//a2 *= a2;

			//result_data[x+y*w] = (a1 * a2 - a3 * a3) - k * SQR(a1 + a2);

			//float e1 = 0.5f * (a1 + a2) + 0.5f * sqrt(4.0f * a3 * a3 + SQR(a1 - a2));
			//float e2 = 0.5f * (a1 + a2) - 0.5f * sqrt(4.0f * a3 * a3 + SQR(a1 - a2));

			//result_data[x+y*w] = e1*e2 - k*SQR(e1+e2);
			
		}
	}

#endif

	time = clock.get();
	std::cout << time << std::endl;


	// copy to output buffer
	unsigned char* out_data = new unsigned char[w * h * 3];
	unsigned int out_size = w * h * 3;

	for (int x = 0; x < w; ++x) {
		for (int y = 0; y < h; ++y) {
#ifdef USE_FLOAT
			out_data[(x+y*w)*3+0] = out_data[(x+y*w)*3+1] = out_data[(x+y*w)*3+2] = (unsigned char)(result_data[x+y*w] * 255.0f);
#else
			out_data[(x+y*w)*3+0] = out_data[(x+y*w)*3+1] = out_data[(x+y*w)*3+2] = result_data[x+y*w];
			//out_data[(x+y*w)*3+0] = in_data[(x+y*w)*4+0];
			//out_data[(x+y*w)*3+1] = in_data[(x+y*w)*4+2];
			//out_data[(x+y*w)*3+2] = in_data[(x+y*w)*4+3];
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
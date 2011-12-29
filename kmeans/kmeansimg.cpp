#include "kmeans.hpp"

#ifdef USE_KMEANS_IMG

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include "windows.h"
#undef max
#undef min
#else
#endif

#include <math.h>
#include <opengl/stb_image.hpp>
#include <iostream>
#include <limits>
#include <util/tostring.hpp>

const int DIM = 3;
typedef Vec<DIM, float> Vecf;
int N = 0;
int K = 0;

Vecf* random_cluster_init(Vecf* input)
{
	int count = 0;
	Vecf* seed = new Vecf[K];

	// find random, but unique centroids
	while (count < K) {
		Vecf* centroid = &input[rand() % N];
		bool found = false;

		for (int i = 0; i < count; ++i) {
			bool equal = true;
			for (int j = 0; j < DIM; ++j) {
				if ((*centroid)[j] != seed[i][j]) {
					equal = false;
					break;
				}
			}
			if (equal) {
				found = true;
				break;
			}
		}

		if (!found) {
			memcpy(seed[count++].v, centroid, DIM * sizeof(float));
		}
	}

	return seed;
}

void cluster_assignment(Vecf* input, Vecf* centroids, int* mapping)
{
	// for each input vector
	for (int i = 0; i < N; ++i) {
		float min_dist = std::numeric_limits<float>::max();

		// for each centroid
		for (int j = 0; j < K; j++) {
			float dist = (input[i] - centroids[j]).lenlen();

			if (dist < min_dist) {
				mapping[i] = j;
				min_dist = dist;
			}
		}
	}
}

void cluster_reposition(Vecf* input, Vecf* centroids, int* mapping)
{
	float* count = new float[K];

	for (int i = 0; i < K; ++i) {
		count[i] = 0.0f;
		for (int j = 0; j < DIM; ++j)
			centroids[i][j] = 0.0f;
	}

	for (int i = 0; i < N; ++i) {
		count[mapping[i]] += 1.0f;
		for (int j = 0; j < DIM; ++j) {
			centroids[mapping[i]][j] += input[i][j];
		}
	}

	for (int i = 0; i < K; ++i)
		for (int j = 0; j < DIM; ++j)
			centroids[i][j] /= count[i];

	delete[] count;
}



int main(int argc, char** argv)
{
	std::string in_file = "in.bmp";
	std::string out_file = "out";

	int w = 0, h = 0, c = 0;
	unsigned char* in_data = stbi_load(in_file.c_str(), &w, &h, &c, STBI_rgb_alpha);
	//float* in_dataf = stbi_loadf(in_file.c_str(), &w, &h, &c, STBI_rgb);

	if (!in_data) {
		std::cout << "Error: Could not read input file" << std::endl;
		return 1;
	}

	N = w * h;

	float* in_dataf = new float[N*DIM];
	for (int i = 0; i < N; ++i) {
		in_dataf[i*3] = ((float)in_data[i*4]) / 255.0f;
		in_dataf[i*3+1] = ((float)in_data[i*4+1]) / 255.0f;
		in_dataf[i*3+2] = ((float)in_data[i*4+2]) / 255.0f;
		//in_dataf[i*5+3] = i % w;
		//in_dataf[i*5+4] = i / w;
	}

	Vecf* input = (Vecf *)in_dataf;
	int* mapping = new int[N];

	for (int num_clusters = 1; num_clusters <= 25; ++num_clusters) { 

		K = num_clusters;


		// first k
		//Vecf* centroids = new Vecf[K];
		//for (int i = 0; i < K; ++i)
		//	memcpy((void*)centroids[i].v, (void*)input[i].v, DIM * sizeof(float));

		Vecf* centroids = random_cluster_init(input);

		for (int i = 0; i < 25; ++i) {
			cluster_assignment(input, centroids, mapping);
			cluster_reposition(input, centroids, mapping);
		}


		// copy to output buffer
		unsigned char* out_data = new unsigned char[w * h * 3];
		unsigned int out_size = w * h * 3;

		for (int x = 0; x < w; ++x) {
			for (int y = 0; y < h; ++y) {
				out_data[(x+y*w)*3+0] = centroids[mapping[x+y*w]][2] * 255.0f;
				out_data[(x+y*w)*3+1] = centroids[mapping[x+y*w]][1] * 255.0f;
				out_data[(x+y*w)*3+2] = centroids[mapping[x+y*w]][0] * 255.0f;
	/*
				out_data[(x+y*w)*3+0] = in_dataf[(x+y*w)*3+0] * 255.0f;
				out_data[(x+y*w)*3+1] = in_dataf[(x+y*w)*3+1] * 255.0f;
				out_data[(x+y*w)*3+2] = in_dataf[(x+y*w)*3+2] * 255.0f;

				out_data[(x+y*w)*3+0] = in_data[(x+y*w)*4+2];
				out_data[(x+y*w)*3+1] = in_data[(x+y*w)*4+1];
				out_data[(x+y*w)*3+2] = in_data[(x+y*w)*4+0];
	*/
			}
		}

		delete[] centroids;

		std::string out_name = out_file + util::toString(K) + ".bmp";
		FILE *out_fdesc = fopen(out_name.c_str(), "wb");
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

	}
	
	delete[] mapping;
	

	std::cin.get();
	return 0;
}

#endif /* USE_KMEANS_IMG */
#include "kmeans.hpp"

#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>

#include <math.h>


#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#define BOOST_FILESYSTEM_VERSION 2
#include <boost/filesystem.hpp>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#ifdef USE_OPENCL
	#include <CL/cl.hpp>
	#include <opencl/oclutil.hpp>
#else
	#include <limits>
#endif

#include <util/tostring.hpp>
#include <util/clock.hpp>

#include <string.h>

#include <opengl/stb_image.hpp>

#ifdef _WIN32
	#include <windows.h>
	#undef max
	#undef min
#else
	#include <sys/time.h>
#endif

#include <png.h>
#include <jpeglib.h>

int DIM = 4;
int N = 1024*1024;
int K = 256;
int ITERATIONS = 20;

int AM_LWS = 256; //8
int RP_LWS = 4; //32
int CT_LWS = 256; //32

int USE_ALL_DEVICES = 1;
int device_count = 1;

std::string input_folder = "images";
std::string output_folder = "output";

std::string uploader = "";

std::vector<std::string> image_list;

#ifdef USE_OPENCL
cl::Platform clPlatform;
std::vector<cl::Device> clDevices;
cl::Context clContext;
std::vector<cl::CommandQueue> clQueues;
cl::Program clProgram;
std::vector<cl::Kernel> clClusterAssignment;
std::vector<cl::Kernel> clClusterReposition;
std::vector<cl::Kernel> clClusterReposition_k;
std::vector<cl::Kernel> clClusterReposition_k_c;
std::vector<cl::Kernel> clComputeCost;
std::vector<cl::Buffer> clInputBuf;
std::vector<cl::Buffer> clCentroidBuf;
std::vector<cl::Buffer> clMappingBuf;
std::vector<cl::Buffer> clReductionBuf;
std::vector<cl::Buffer> clConvergedBuf;
#endif


boost::mt19937 rng;
boost::uniform_real<float> u;
boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >* gen;

#ifndef _WIN32
uint64_t getTimeMs(void)
{
    struct timeval tv;

    gettimeofday(&tv, 0);
    return uint64_t( tv.tv_sec ) * 1000 + tv.tv_usec / 1000;
}
#endif

float gen_random_float()
{
    return (*gen)();
}

bool get_image_files(std::string folder)
{
	using namespace boost::filesystem;

	path p (folder);

	if (is_directory(p)) {
		if(!is_empty(p)) {
			directory_iterator end_itr;
			for (directory_iterator itr(p); itr != end_itr; ++itr) {
				image_list.push_back(folder + "/" + itr->leaf());
			}
		}
	}
	return image_list.size() > 0;
}

#ifdef USE_OPENCL
void initCL()
{
	ocl::createContextEx(CL_DEVICE_TYPE_ALL, clPlatform, clDevices, clContext, clQueues);
	cl_int clError = CL_SUCCESS;

	std::ifstream t("kmeans.cl");
	std::string code((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

	std::string header = "#define DIM ";
	header +=  util::toString(DIM);
	header += "\n";
	header += "#define K ";
	header += util::toString(K);
	header += "\n";
	header += "#define N ";
	header += util::toString(N);
	header += "\n";
	header += "#define AM_LWS ";
	header += util::toString(AM_LWS);
	header += "\n";
	header += "#define RP_LWS ";
	header += util::toString(RP_LWS);
	header += "\n\n\n";

	code = header + code;

	try {
		cl::Program::Sources source(1, std::make_pair(code.c_str(), code.size()));
		clProgram = cl::Program(clContext, source);
		clProgram.build(clDevices, "-cl-fast-relaxed-math -cl-unsafe-math-optimizations -cl-mad-enable");

		std::string info("");
		for (std::vector<cl::Device>::iterator itr = clDevices.begin(); itr != clDevices.end(); ++itr) {
			clProgram.getBuildInfo(*itr, CL_PROGRAM_BUILD_LOG, &info);
			if (info.size() > 0)
				std::cout << "Build log: " << info << std::endl;
		}

		for (int i = 0; i < clDevices.size(); ++i) {
			clClusterAssignment.push_back(cl::Kernel(clProgram, "cluster_assignment", &clError));
			clClusterReposition.push_back(cl::Kernel(clProgram, "cluster_reposition", &clError));
			clClusterReposition_k.push_back(cl::Kernel(clProgram, "cluster_reposition_k", &clError));
			clClusterReposition_k_c.push_back(cl::Kernel(clProgram, "c_cluster_reposition", &clError));
			clComputeCost.push_back(cl::Kernel(clProgram, "compute_cost", &clError));
		}

	} catch (const cl::Error& err) {
		std::cout << "OpenCL Error 4: " << err.what() << " (" << err.err() << ")" << std::endl;
		std::string info("");
		for (std::vector<cl::Device>::iterator itr = clDevices.begin(); itr != clDevices.end(); ++itr) {
			clProgram.getBuildInfo(*itr, CL_PROGRAM_BUILD_LOG, &info);
			if (info.size() > 0)
				std::cout << "Build log: " << info << std::endl;
		}
		std::cin.get();
	}
}
#endif

float* random_cluster_init(float* input)
{
	int count = 0;
	float* seed = new float[K*DIM];

	// find random, but unique centroids
	while (count < K) {
		float* centroid = &input[(rand() % N) * DIM];
		bool found = false;

		for (int i = 0; i < count; ++i) {
			bool equal = true;
			for (int j = 0; j < DIM; ++j) {
				if (centroid[j] != seed[i*DIM+j]) {
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
			memcpy(&seed[(count++)*DIM], centroid, DIM * sizeof(float));
		}
	}

	return seed;
}

#ifndef USE_OPENCL
void cluster_assignment(float* input, float* centroids, int* mapping)
{
	// for each input vector
	for (int i = 0; i < N; ++i) {
		float min_dist = std::numeric_limits<float>::max();
		// for each centroid
		for (int j = 0; j < K; j++) {

			float dist = 0.0f;
			for (int l = 0; l < DIM; ++l) {
				dist += (input[i*DIM+l] - centroids[j*DIM+l]) * (input[i*DIM+l] - centroids[j*DIM+l]);
			}
			dist = sqrtf(dist);

			if (dist < min_dist) {
				mapping[i] = j;
				min_dist = dist;
			}
		}
	}

}

void cluster_reposition(float* input, float* centroids, int* mapping)
{
	float* count = new float[K];

	for (int i = 0; i < K; ++i) {
		count[i] = 0.0f;
		for (int j = 0; j < DIM; ++j)
			centroids[i*DIM+j] = 0.0f;
	}

	for (int i = 0; i < N; ++i) {
		count[mapping[i]] += 1.0f;
		for (int j = 0; j < DIM; ++j) {
			centroids[mapping[i]*DIM+j] += input[i*DIM+j];
		}
	}

	for (int i = 0; i < K; ++i)
		for (int j = 0; j < DIM; ++j)
			centroids[i*DIM+j] /= count[i];

	delete[] count;
}
#endif

void save_image(const std::string& file, const int w, const int h, float* centroids, int* mapping)
{
	boost::filesystem::path path(file);
	std::string out_name = output_folder + "/" + boost::filesystem::basename(file);

	// copy to output buffer
	unsigned char* out_data = new unsigned char[w * h * 3];
	unsigned int out_size = w * h * 3;

	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			out_data[(x+y*h)*3+0] = (unsigned char)(centroids[mapping[x+y*h]*DIM+0] * 255.0f);
			out_data[(x+y*h)*3+1] = (unsigned char)(centroids[mapping[x+y*h]*DIM+1] * 255.0f);
			out_data[(x+y*h)*3+2] = (unsigned char)(centroids[mapping[x+y*h]*DIM+2] * 255.0f);
		}
	}

	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;

	FILE * outfile;
	JSAMPROW* row_pointer = new JSAMPROW[h];
	int row_stride;

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);

	if ((outfile = fopen(out_name.c_str(), "wb")) == NULL) {
		return;
	}

	jpeg_stdio_dest(&cinfo, outfile);

	cinfo.image_width = w;
	cinfo.image_height = h;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;

	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, 80, TRUE);
	jpeg_start_compress(&cinfo, TRUE);

	row_stride = w * 3;
	for (int k = 0; k < h; k++)
		row_pointer[k] = (JSAMPROW) (out_data + k*w*3);

	jpeg_write_scanlines(&cinfo, row_pointer, h);


	jpeg_finish_compress(&cinfo);
	fclose(outfile);
	jpeg_destroy_compress(&cinfo);
	delete[] row_pointer;

	/*

	FILE *fp = NULL;
	png_structp png_ptr;
	png_infop info_ptr;
	png_colorp palette;
	png_bytep trans;

	fp = fopen(out_name.c_str(), "wb");
	if (fp == NULL)
		return;

   png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (png_ptr == NULL) {
		fclose(fp);
		return;
	}

	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		fclose(fp);
		png_destroy_write_struct(&png_ptr,  NULL);
		return;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		fclose(fp);
		png_destroy_write_struct(&png_ptr, &info_ptr);
		return;
	}

	png_init_io(png_ptr, fp);

	png_set_IHDR(png_ptr, info_ptr, w, h, 8, PNG_COLOR_TYPE_PALETTE,
		PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	std::cout << "setihdr" << std::endl;

	palette = (png_colorp)png_malloc(png_ptr, K * png_sizeof(png_color));

	std::cout << "png_malloc" << std::endl;

	for (int i = 0; i < K; ++i) {
		//palette[i].red = in_dataf[i*4]*255;
		//palette[i].green = in_dataf[i*4+1]*255;
		//palette[i].blue = in_dataf[i*4+2]*255;
		palette[i].red = (unsigned char)(centroids[i*DIM+0]*255.0f);
		palette[i].green = (unsigned char)(centroids[i*DIM+1]*255.0f);
		palette[i].blue = (unsigned char)(centroids[i*DIM+2]*255.0f);
	}

	std::cout << "gen palette" << std::endl;
	png_set_PLTE(png_ptr, info_ptr, palette, K);

	std::cout << "set palette" << std::endl;

	trans = (png_bytep)png_malloc(png_ptr, K * png_sizeof(png_byte));
	for (int i = 0; i < K; ++i)
		trans[i] = (unsigned char)(centroids[i*DIM+3]*255.0f);

	std::cout << "gen trans" << std::endl;

	png_set_tRNS(png_ptr, info_ptr, trans, K, NULL);

	std::cout << "set trans" << std::endl;

	png_write_info(png_ptr, info_ptr);

	std::cout << "write info" << std::endl;

	int number_passes = 1;

	png_uint_32 k;
	png_byte* image = new png_byte[h*w*1];

	std::cout << "alloc image" << std::endl;

	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			image[y*w+x] = (unsigned char)mapping[x+y*w];
		}
	}

	std::cout << "gen image" << std::endl;

	png_bytep* row_pointers = new png_bytep[h];

	for (k = 0; k < h; k++)
		row_pointers[k] = (png_bytep) (image + k*w*1);

	std::cout << "gen pointer" << std::endl;

	png_write_image(png_ptr, row_pointers);

	std::cout << "write image" << std::endl;

	png_write_end(png_ptr, info_ptr);

	png_free(png_ptr, palette);
	palette = NULL;

	png_free(png_ptr, trans);
	trans = NULL;

	png_destroy_write_struct(&png_ptr, &info_ptr);

	fclose(fp);

	delete[] row_pointers;
	delete[] image;

*/
	/*

	// copy to output buffer
	unsigned char* out_data = new unsigned char[w * h * 3];
	unsigned int out_size = w * h * 3;

	for (int x = 0; x < w; ++x) {
		for (int y = 0; y < h; ++y) {
			out_data[(x+y*w)*3+0] = centroids[mapping[x+y*w]*DIM+2] * 255.0f;
			out_data[(x+y*w)*3+1] = centroids[mapping[x+y*w]*DIM+1] * 255.0f;
			out_data[(x+y*w)*3+2] = centroids[mapping[x+y*w]*DIM+0] * 255.0f;
		}
	}


	FILE *out_fdesc = fopen(out_name.c_str(), "wb");
	if (out_fdesc == NULL) {
		std::cerr << "Error: Could not create output file" << std::endl;
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
*/

	delete[] centroids;
	delete[] mapping;

	if (uploader.size() > 0) {
		std::string command = uploader + " " + out_name;
		std::cout << "Uploading: " << command << std::endl;
		system(command.c_str());
	}
}

void prefetch_image(const std::string& file, int* w, int* h, float** input, float** centroids, int** mapping, bool* success, bool* fetched)
{
	int c = 0;
	*input = stbi_loadf(file.c_str(), w, h, &c, STBI_rgb_alpha);
	if (!(*input)) {
		std::cerr << "Could not load " << file << std::endl;
		*success = false;
		if (fetched)
			*fetched = true;
		return;
	}

	*centroids = random_cluster_init(*input);
	*mapping = new int[N];
	*success = true;
	if (fetched)
		*fetched = true;
}

void exec(int id, bool threaded)
{
	if (threaded)
		std::cout << "thread:" << boost::this_thread::get_id() << ": " << id << std::endl;

#ifdef USE_OPENCL
	const cl::CommandQueue& queue = clQueues[id];
	const cl::Kernel& assignment = clClusterAssignment[id];
	const cl::Kernel& reposition = clClusterReposition[id];
	const cl::Kernel& reposition_k = clClusterReposition_k[id];
	const cl::Kernel& reposition_k_c = clClusterReposition_k_c[id];
	const cl::Kernel& cost = clComputeCost[id];

	const cl::Buffer& inputBuf = clInputBuf[id];
	const cl::Buffer& mappingBuf = clMappingBuf[id];
	const cl::Buffer& centroidBuf = clCentroidBuf[id];
	//const cl::Buffer& convergedBuf = clConvergedBuf[id];
	const cl::Buffer& reductionBuf = clReductionBuf[id];

	int _w = 0, _h = 0;
	bool _success = false;
	float* _input = NULL;
	float* _centroids = NULL;
	int* _mapping = NULL;
	bool _fetched = false;
#endif

	int iteration = 0;

	for (int pass = id; pass < image_list.size(); pass += device_count) {
		std::string& file = image_list[pass];

		int w = 0, h = 0, c = 0;
		float* input = NULL;
		float* centroids = NULL;
		int* mapping = NULL;
		bool success = false;

#ifdef USE_OPENCL
		if (pass == id) {
			prefetch_image(file, &w, &h, &input, &centroids, &mapping, &success, NULL);
			if (!success) {
				_fetched = false;
				boost::thread worker(boost::bind(prefetch_image, image_list[pass + device_count], &_w, &_h, &_input, &_centroids, &_mapping, &_success, &_fetched));
				continue;
			}
		} else {
			while (!_fetched)
				;
			//prefetch_image(file, &w, &h, &input, &centroids, &mapping, &success, NULL);
			if (!_success) {
				_fetched = false;
				boost::thread worker(boost::bind(prefetch_image, image_list[pass + device_count], &_w, &_h, &_input, &_centroids, &_mapping, &_success, &_fetched));
				continue;
			}

			w = _w;
			h = _h;
			input = _input;
			centroids = _centroids;
			mapping = _mapping;
		}

		if (pass + device_count < image_list.size()) {
			_fetched = false;
			boost::thread worker(boost::bind(prefetch_image, image_list[pass + device_count], &_w, &_h, &_input, &_centroids, &_mapping, &_success, &_fetched));
			//prefetch_image(image_list[pass + device_count], &_w, &_h, &_input, &_centroids, &_mapping, &_success, &_fetched);
		}
#else
		prefetch_image(file, &w, &h, &input, &centroids, &mapping, &success, NULL);
		if (!success)
			continue;
#endif


		util::Clock clock;
		clock.reset();

#ifdef USE_OPENCL

		queue.enqueueWriteBuffer(inputBuf, CL_FALSE, 0, N * DIM * sizeof(float), (void*)input, NULL, NULL);
		queue.enqueueWriteBuffer(centroidBuf, CL_FALSE, 0, K * DIM * sizeof(float), (void*)centroids, NULL, NULL);

		for (int i = 0; i < ITERATIONS; ++i) {
			queue.enqueueNDRangeKernel(assignment, cl::NullRange, cl::NDRange(N), cl::NDRange(AM_LWS), NULL, NULL);

#ifdef CLUSTER_REPOSITION_K
			queue.enqueueNDRangeKernel(reposition_k, cl::NullRange, cl::NDRange(K), cl::NDRange(RP_LWS), NULL, NULL);
#else
			queue.enqueueNDRangeKernel(reposition, cl::NullRange, cl::NDRange(DIM), cl::NDRange(RP_LWS), NULL, NULL);
#endif
		}

		//queue.enqueueNDRangeKernel(cost, cl::NullRange, cl::NDRange(N), cl::NDRange(CT_LWS), NULL, NULL);

		//queue.finish();



		queue.enqueueReadBuffer(centroidBuf, CL_FALSE, 0, K * DIM * sizeof(float), centroids, NULL, NULL);
		queue.enqueueReadBuffer(mappingBuf, CL_TRUE, 0, N * sizeof(int), mapping, NULL, NULL);
		//queue.enqueueReadBuffer(reductionBuf, CL_FALSE, 0, N * sizeof(float), treduction, NULL, NULL);


		//queue.finish();
#else
		for (int i = 0; i < ITERATIONS; ++i) {
			cluster_assignment(input, centroids, mapping);
			cluster_reposition(input, centroids, mapping);
		}
#endif

		float now = clock.get();
		std::cout << "Device: " << now << std::endl;

#ifdef USE_OPENCL
		boost::thread worker(boost::bind(save_image, file, w, h, centroids, mapping));
#else
		save_image(file, w, h, centroids, mapping);
#endif

		free(input);

	}
/*
	float sum = 0.0f;
	for (int i = 0; i < N; ++i) {
		sum += treduction[i];
	}

	sum /= (float)N;

	std::cout << sum << std::endl;
*/
}

int main(int argc, char** argv)
{
	int args = 1;
#ifdef USE_OPENCL
	if (argc < 9) {
#else
	if (argc < 6) {
#endif
		std::cout << "Not enough arguments" << std::endl;
		system("pause");
		return 1;
	}

	DIM = util::toInt(argv[args++]);
	N = util::toInt(argv[args++]); N*=N;
	K = util::toInt(argv[args++]);
	ITERATIONS = util::toInt(argv[args++]);
#ifdef USE_OPENCL
	AM_LWS = util::toInt(argv[args++]);
	RP_LWS = util::toInt(argv[args++]);
	CT_LWS = util::toInt(argv[args++]);

	USE_ALL_DEVICES = util::toInt(argv[args++]);
#else
	device_count = util::toInt(argv[args++]);
#endif

	if (args < argc) {
		input_folder = argv[args++];
		if (args < argc) {
			output_folder = argv[args++];
			if (args < argc) {
				uploader = argv[args++];
			}
		}
	}

	std::cout << "DIM = " << DIM << std::endl;
	std::cout << "N = " << N << std::endl;
	std::cout << "K = " << K << std::endl;
	std::cout << "ITERATIONS = " << ITERATIONS << std::endl;
#ifdef USE_OPENCL
	std::cout << "AM_LWS = " << AM_LWS << std::endl;
	std::cout << "RP_LWS = " << RP_LWS << std::endl;
	std::cout << "CT_LWS = " << CT_LWS << std::endl;
	std::cout << "USE_ALL_DEVICES = " << USE_ALL_DEVICES << std::endl << std::endl;
#else
	std::cout << "device_count = " << device_count << std::endl << std::endl;
#endif
	std::cout << "input_folder = " << input_folder << std::endl;
	std::cout << "output_folder = " << output_folder << std::endl;
	std::cout << "uploader = " << uploader << std::endl;


	if (!get_image_files(input_folder)) {
		std::cerr << "Could not find images" << std::endl;
		return 1;
	} else {
		std::cout << "IMAGE_COUNT = " << image_list.size() << std::endl;
	}

#ifdef _WIN32
	rng.seed();
	srand(GetTickCount());
#else
	rng.seed();
	srand(getTimeMs());
#endif

	u = boost::uniform_real<float>(0.0f, 256.0f);
	gen = new boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >(rng, u);

#ifdef USE_OPENCL
	cl_int clError = CL_SUCCESS;
	initCL();

	for (int i = 0; i < clDevices.size(); ++i) {

		clInputBuf.push_back(cl::Buffer(clContext, CL_MEM_READ_ONLY, N * DIM * sizeof(float), NULL, &clError));
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

		clCentroidBuf.push_back(cl::Buffer(clContext, CL_MEM_READ_WRITE, K * DIM * sizeof(float), NULL, &clError));
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

		clMappingBuf.push_back(cl::Buffer(clContext, CL_MEM_READ_WRITE, N * sizeof(int), NULL, &clError));
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

		clReductionBuf.push_back(cl::Buffer(clContext, CL_MEM_WRITE_ONLY, N * sizeof(int), NULL, &clError));
		if (clError != CL_SUCCESS) std::cout << "OpenCL Error: Could not create buffer" << std::endl;

		clClusterAssignment[i].setArgs(clInputBuf[i](), clCentroidBuf[i](), clMappingBuf[i]());
		clClusterReposition[i].setArgs(clInputBuf[i](), clMappingBuf[i](), clCentroidBuf[i]());
		clClusterReposition_k[i].setArgs(clInputBuf[i](), clMappingBuf[i](), clCentroidBuf[i]());
		//clClusterReposition_k_c[i].setArgs(clInputBuf[i](), clMappingBuf[i](), clCentroidBuf[i](), clConvergedBuf[i]());
		clComputeCost[i].setArgs(clInputBuf[i](), clCentroidBuf[i](), clMappingBuf[i](), clReductionBuf[i]());

	}

	device_count = clDevices.size();
#endif

	util::Clock clock;
	clock.reset();

	if (device_count > 1) {
		boost::thread_group threads;

		for (int i = 0; i < device_count; ++i) {
			threads.create_thread(boost::bind(exec, i, true));
		}

		threads.join_all();
	} else {
		exec(0, false);
	}

	float now = clock.get();
	std::cout << "Total: " << now << std::endl;

	system("pause");

	return 0;
}

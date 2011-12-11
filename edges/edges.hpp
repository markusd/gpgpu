#ifndef EDGES_HPP_
#define EDGES_HPP_

#define USE_CUDA
#define USE_SHARED
//#define USE_FLOAT

#ifdef USE_FLOAT
	#define PIXEL float
#else
	#define PIXEL unsigned char
#endif

#endif
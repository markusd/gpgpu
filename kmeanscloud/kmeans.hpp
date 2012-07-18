#ifndef KMEANS_HPP_
#define KMEANS_HPP_


#define CLUSTER_REPOSITION_K

const int DIM = 4;
const int N = 1024*1024;
const int K = 256;
const int ITERATIONS = 20;

const int AM_LWS = 256; //8
const int RP_LWS = 4; //32


/*
const int DIM = 1024;
const int N = 4096;
const int K = 16;
const int ITERATIONS = 20;

const int AM_LWS = 8;
const int RP_LWS = 64; //32
*/

#define __CL_ENABLE_EXCEPTIONS

#endif
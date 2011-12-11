#ifndef KMEANS_HPP_
#define KMEANS_HPP_

#include <m3d/m3d.hpp>
#include <vector>
#include <map>

#define __CL_ENABLE_EXCEPTIONS
//#define USE_VISUALIZATION

using namespace m3d;

typedef enum SeedingAlgorithm { 
	RANDOM,
	MANUAL,
	HARTIGAN_WONG
};

/* distance functions */
typedef double (*DistanceFunction)(Vec2d a, Vec2d b);

double euclidian_distance(Vec2d a, Vec2d b); 


/* Seeding algorithms */
std::vector<Vec2d> random_seed(unsigned int k, const std::vector<Vec2d>& input);
std::pair<Vec2d, std::vector<Vec2d> > hartigan_wong(unsigned int k, const std::vector<Vec2d>& input);

/**
 * kmeans algorithm
 *
 * returns (centroids[], cost)
 */
std::pair<std::vector<Vec2d>, double> kmeans(unsigned int iterations, unsigned int k, const std::vector<Vec2d>& input, const std::vector<Vec2d>& seed);

#endif
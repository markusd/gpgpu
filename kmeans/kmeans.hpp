#ifndef KMEANS_HPP_
#define KMEANS_HPP_

#include <m3d/m3d.hpp>
#include <vector>
#include <map>

using namespace m3d;

//#define INPUT_SIZE 512
//#define ITER_MAX 100

typedef enum SeedingAlgorithm { 
	RANDOM,
	MANUAL,
	HARTIGAN_WONG
};

/* Seeding algorithms */
std::vector<Vec2d> random_seed(unsigned int k, const std::vector<Vec2d>& input);
std::pair<Vec2d, std::vector<Vec2d> > hartigan_wong(unsigned int k, const std::vector<Vec2d>& input);

/* kmeans */
std::vector<Vec2d> kmeans(unsigned int iterations, unsigned int k, const std::vector<Vec2d>& input, const std::vector<Vec2d>& seed);

#endif
#ifndef KMEANS_HPP_
#define KMEANS_HPP_

#include <m3d/m3d.hpp>
#include <vector>
#include <map>

using namespace m3d;

#define INPUT_SIZE 512
#define ITER_MAX 10

std::vector<Vec2d> kmeans(unsigned int k, const std::vector<Vec2d>& input);

#endif
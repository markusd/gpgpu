#define WIN32_LEAN_AND_MEAN
#include "windows.h"

#include "kmeans.hpp"

#include <iostream>
#include <limits>
#undef max

#include <mainwindow.hpp>

#include <QtGui/QApplication>

using namespace m3d;

std::vector<Vec2d> kmeans(unsigned int k, const std::vector<Vec2d>& input)
{
	std::vector<Vec2d> seed;
	std::vector<std::pair<Vec2d, unsigned int> > new_centroids;

	// input-id --> centroid-id
	std::map<unsigned int, unsigned int> dist_mapping;

	// get k random vectors as seed
	for (int i = 0; i < k; i++) {
		seed.push_back(input[i]);
	}

	for (int iter = 0; iter < ITER_MAX; ++iter) {

		// find closest centroid for each input vector
		for (int i = 0; i < input.size(); ++i) {

			double min_dist = std::numeric_limits<double>::max();

			for (int j = 0; j < k; j++) {
				double dist = (input[i] - seed[j]).len();
				if (dist < min_dist) {
					dist_mapping[i] = j;
					min_dist = dist;
				}
			}
		}

		// initialize new centroids
		new_centroids.clear();
		for (int i = 0; i < k; ++i)
			new_centroids.push_back(std::make_pair(Vec2d(0.0, 0.0), 0));


		for (std::map<unsigned int, unsigned int>::iterator itr = dist_mapping.begin();
			itr != dist_mapping.end(); ++itr) {
				new_centroids[itr->second].first += input[itr->first];
				++new_centroids[itr->second].second;
		}

		for (int i = 0; i < k; ++i) {
			seed[i] = new_centroids[i].first * (1.0 / (double)new_centroids[i].second);
		}

	}

	return seed;
}

int main(int argc, char** argv)
{
	QApplication app(argc, argv);
	MainWindow mainwindow(&app);

	int result = app.exec();
	return result;
}
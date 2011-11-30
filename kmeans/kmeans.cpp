#define WIN32_LEAN_AND_MEAN
#include "windows.h"

#include "kmeans.hpp"

#include <iostream>
#include <limits>
#undef max

#include <mainwindow.hpp>

#include <QtGui/QApplication>

using namespace m3d;

std::vector<Vec2d> random_seed(unsigned int k, const std::vector<Vec2d>& input)
{
	std::vector<Vec2d> seed;

	// find random, but unique centroids
	while (seed.size() < k) {
		Vec2d centroid = input[rand() % input.size()];
		if (std::find(seed.begin(), seed.end(), centroid) == seed.end())
			seed.push_back(centroid);
	}

	return seed;
}

// compare distance of vectors to compare_mean
Vec2d compare_mean;
bool compare_dist(Vec2d i, Vec2d j)
{
	return ((i - compare_mean).lenlen() < (j - compare_mean).lenlen());
}

std::pair<Vec2d, std::vector<Vec2d> > hartigan_wong(unsigned int k, const std::vector<Vec2d>& input)
{
	std::vector<Vec2d> seed;
	Vec2d mean(0.0, 0.0);

	for (int i = 0; i < input.size(); ++i) {
		mean += input[i];
	}
	mean *= (1.0 / (double)input.size());

	std::vector<Vec2d> sorted(input);

	compare_mean = mean;
	std::sort(sorted.begin(), sorted.end(), compare_dist);

	//for (int i = 0; i < sorted.size(); ++i)
	//	std::cout << (sorted[i] - mean).len() << std::endl;
	//std::cout << std::endl;

	unsigned int gap = (sorted.size() - 1) / std::max<unsigned int>(k - 1, 1);
	//unsigned int gap = sorted.size() / std::max<unsigned int>(k - 1, 1);
	//if (gap * (k-1) == sorted.size())
	//	--gap;

	for (int i = 0; i < k; ++i)
		seed.push_back(sorted[i*gap]);

	//for (int i = 0; i < seed.size(); ++i)
	//	std::cout << (seed[i] - mean).len() << std::endl;

	return std::make_pair(mean, seed);
}


std::vector<Vec2d> kmeans(unsigned int iterations, unsigned int k, const std::vector<Vec2d>& input, const std::vector<Vec2d>& seed)
{
	std::vector<Vec2d> centroids;
	std::vector<std::pair<Vec2d, unsigned int> > new_centroids;

	// input-id --> centroid-id
	std::map<unsigned int, unsigned int> dist_mapping;

	// initialize centroids
	for (int i = 0; i < k; i++) {
		centroids.push_back(seed[i]);
	}

	for (int iter = 0; iter < iterations; ++iter) {

		// find closest centroid for each input vector
		for (int i = 0; i < input.size(); ++i) {

			double min_dist = std::numeric_limits<double>::max();

			for (int j = 0; j < k; j++) {
				double dist = (input[i] - centroids[j]).len();
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
			if (new_centroids[i].second == 0)
				std::cout << "Cluster is empty" << std::endl;
			double weight = new_centroids[i].second ? (double)new_centroids[i].second : 1.0;
			centroids[i] = new_centroids[i].first * (1.0 / weight);
		}

	}

	return centroids;
}

int main(int argc, char** argv)
{
	srand(GetTickCount());

	QApplication app(argc, argv);
	MainWindow mainwindow(&app);

	int result = app.exec();
	return result;
}
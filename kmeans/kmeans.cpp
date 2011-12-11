#define WIN32_LEAN_AND_MEAN
#include "windows.h"

#include "kmeans.hpp"

#include <iostream>
#include <limits>
#undef max

#include <mainwindow.hpp>

#include <QtGui/QApplication>

using namespace m3d;

/* distance functions */
double euclidian_distance(Vec2d a, Vec2d b)
{
	return (a - b).len();
}

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

bool second_sort(std::pair<unsigned int, unsigned int> p1, std::pair<unsigned int, unsigned int> p2) {
	return p1.second > p2.second;
}

std::vector<Vec2d> astrahan(unsigned int k, const std::vector<Vec2d>& input) {
	// vec id - density
	std::vector<std::pair<unsigned int, unsigned int> > densities;
	//vec id, vec id - distance
	std::vector<std::vector<double> > distances;

	distances.reserve(input.size());
	distances.resize(input.size());

	// initialize
	for(int h = 0; h < input.size(); ++h) {
		densities.push_back(std::make_pair(h, 0));
	}

	// get all distance pairs
	double distance_mean = 0.0;
	for(int i = 0; i < input.size(); ++i) {
		distances[i].reserve(input.size());
		distances[i].resize(input.size());
		for(int j = 0; j < input.size(); ++j){
			distances[i][j] = euclidian_distance(input[i],input[j]);
			distance_mean += distances[i][j];
		}
	}
	
	distance_mean = distance_mean/(input.size()*input.size());

	for(int i = 0; i < input.size(); ++i){
		for(int j = 0; j < input.size(); ++j) {
			if(distances[i][j] <= distance_mean)
					densities[i].second++;
		}
	}

	printf("%f\n", distance_mean);
	
	std::sort(densities.begin(), densities.end(), second_sort);
	
	std::vector< std::pair<unsigned int, unsigned int> > tmp_result;
	tmp_result.push_back(densities.front());

	for(int i = 1; i < densities.size(); ++i){
		bool inside = true;
		for(int j = 0; j < tmp_result.size(); ++j) {
			if(euclidian_distance(input[densities[i].first], input[tmp_result[j].first]) > distance_mean) {
				inside = false;
			} else {
				inside = true;
				break;
			}
		}
		if(!inside)
			tmp_result.push_back(densities[i]);
		if(tmp_result.size() == k)
			break;
	}

	std::vector<Vec2d> result;// = random_seed(k, input);
	if(tmp_result.size() == k){
		for(int i = 0; i < k; ++i){
			result.push_back(input[tmp_result[i].first]);
		}
	} else {
		for(int i = 0; i < tmp_result.size(); ++i) {
			result.push_back(input[tmp_result[i].first]);
		}
		while(result.size() < k) {
			result.push_back(input[densities.back().first]);
			densities.pop_back();
		}
	}

	return result;
}

std::pair<std::vector<Vec2d>, double> kmeans(unsigned int iterations, unsigned int k, const std::vector<Vec2d>& input, const std::vector<Vec2d>& seed)
{
	std::vector<Vec2d> centroids;

	// centroid vector --> number of assigned input vectors
	std::vector<std::pair<Vec2d, unsigned int> > new_centroids;

	// input-id --> centroid-id
	std::map<unsigned int, unsigned int> dist_mapping;

	// initialize centroids
	for (int i = 0; i < k; i++) {
		centroids.push_back(seed[i]);
	}

	for (int iter = 0; iter < iterations; ++iter) {

		double cost = 0.0;

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

			cost += (input[i] - centroids[dist_mapping[i]]).lenlen();
		}

		// normalize cost
		cost /= (double)input.size();
		std::cout << "Iter " << iter << ": cost = " << cost << std::endl;

		// initialize new centroids
		new_centroids.clear();
		for (int i = 0; i < k; ++i)
			new_centroids.push_back(std::make_pair(Vec2d(0.0, 0.0), 0));


		// compute cumulated centroids position and the number of assigned input vectors
		for (std::map<unsigned int, unsigned int>::iterator itr = dist_mapping.begin();
			itr != dist_mapping.end(); ++itr) {
				new_centroids[itr->second].first += input[itr->first];
				++new_centroids[itr->second].second;
		}

		// weight (normalize) the new centroids
		for (int i = 0; i < k; ++i) {
			if (new_centroids[i].second == 0)
				std::cout << "Cluster is empty" << std::endl;
			double weight = new_centroids[i].second ? (double)new_centroids[i].second : 1.0;
			centroids[i] = new_centroids[i].first * (1.0 / weight);
		}

	}


	// compute final cost

	double cost = 0.0;

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

		cost += (input[i] - centroids[dist_mapping[i]]).lenlen();
	}

	// normalize cost
	cost /= (double)input.size();
	std::cout << "Final cost = " << cost << std::endl;

	return std::make_pair(centroids, cost);
}

#ifdef USE_VISUALIZATION

int main(int argc, char** argv)
{
	srand(GetTickCount());

	QApplication app(argc, argv);
	MainWindow mainwindow(&app);

	int result = app.exec();
	return result;
}

#endif /* USE_VISUALIZATION */

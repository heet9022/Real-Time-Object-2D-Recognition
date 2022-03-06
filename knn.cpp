#include "knn.h"
#include "csv_helper.h"

class Data {
public:
	double distance;
	string label;
	vector<double> features;
	Data() {
		distance = 0;
	}
};

//double euclideanDistance(Company& lhs, Company& test) {
	//return sqrt(pow((lhs.turnover - test.turnover), 2) + pow((lhs.characteristics - test.characteristics), 2));
//}

bool cmp(Data& a, Data& b) {
	return a.distance < b.distance;
}

double manhattanDistance(vector<double>& f1, vector<double>& f2) {

	double sum = 0;

	for (int i = 0; i < f1.size(); i++) {
		sum += abs(f1[i] - f2[i]);
	}
	return sum;
}

void fillDistances(vector<vector<double>> &query, vector<vector<double>> &nfeatures, vector<string> &labels, vector<Data> &data) {
	for (int i = 0; i < labels.size(); i++) {

		Data data_point;
		data_point.label = labels[i];
		data_point.features = nfeatures[i];
		data_point.distance = manhattanDistance(data_point.features, query[0]);
		data.push_back(data_point);
	}
}

string KNN(vector<vector<double>>& query, int k) {

	string fileName = "training_feature_database.csv";
	std::vector<std::string> labels;
	std::vector<std::vector<double>> nfeatures;
	readFromFile(fileName, labels, nfeatures);

	vector<Data> data;

	//filling the distances between all points and test
	fillDistances(query, nfeatures, labels, data);

	//sorting so that we can get the k nearest
	sort(data.begin(), data.end(), cmp);

	map<string, int> count;
	int max = -1;
	string mode_label;

	for (int i = 0; i < k; i++) {
		count[data[i].label] += 1;
		if (count[data[i].label] > max) {
			max = count[data[i].label];
			mode_label = data[i].label;
		}
	}

	return mode_label;
}

#include "csv_helper.h"

/*
* Appends a row to CSV
* std::string file_name: File name of the csv file
* std::string label: Label for the features
* std::vector<double> features: Features to be appended
*/

bool writeToFile(std::string file_name, std::string label, std::vector<double> features) {

	std::ofstream file;
	file.open(file_name, std::ios::app);
	file << label << ",";
	for (double feature : features)
		file << feature << ",";
	file << std::endl;

	file.close();

	return true;
}

/*
* Writes a confusion matrix to csv
* std::string file_name: File name of the csv file
* set<string> &s: Set of labels
*  map<string, map<string, int>> &cm: Consfusion matrix
*/

bool writeConfusionMatrix(string file_name, set<string> &s, map<string, map<string, int>> &cm) {

	vector<string> labels(s.size());
	std::copy(s.begin(), s.end(), labels.begin());
	std::ofstream file;
	file.open(file_name, std::ios::app);
	file << "*,";
	for (string label : labels)
		file << label << ",";
	file << endl;
	for (string pred_label : labels) {
		file << pred_label << ",";
		for (string real_label : labels) {
			file << cm[pred_label][real_label] << ",";
		}
		file << endl;
	}

	file.close();

	return true;

}

/*
* Read from a CSV file
* std::string file_name: File name of the csv file
* std::vector<std::string> &labels: Number of labels
* std::vector<std::vector<double>> &nfeatures: Trained features
*/


bool readFromFile(std::string file_name, std::vector<std::string> &labels, std::vector<std::vector<double>> &nfeatures) {

	int size = 7;
	std::ifstream file;
	file.open(file_name);
	std::string label;
	std::string element;
	while (getline(file, label, ',')) {

		std::vector<double> features;

		for (int i = 0; i < size - 1; i++) {
			getline(file, element, ',');
			features.push_back(std::stod(element));
		}
		getline(file, element, '\n');
		features.push_back(std::stod(element));
		nfeatures.push_back(features);
		labels.push_back(label);
	}

	file.close();

	return true;
}



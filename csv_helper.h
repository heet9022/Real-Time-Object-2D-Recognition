#pragma once
class csv_helper
{
};

#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
using namespace std;


bool writeToFile(std::string file_name, std::string label, std::vector<double> features);
bool writeConfusionMatrix(string file_name, set<string>& labels, map<string, map<string, int>>& cm);
bool readFromFile(std::string file_name, std::vector<std::string>& labels, std::vector<std::vector<double>>& nfeatures);

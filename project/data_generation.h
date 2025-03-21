#include <bits/stdc++.h>
#include <glpk.h>
using namespace std;
vector<vector<double>> generate_random_g(int n, int k, int m);
double solve(int k, int n, const vector<vector<double>>& G, int m);
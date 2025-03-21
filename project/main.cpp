#include "data_generation.h"
//n ∈ {9, 10}, k ∈ {4, 5, 6}, m ∈ {2, 3, · · · , n − k}
using std::cout;
int main(){
    int n = 9;
    int k = 4;
    int m = 2;
    vector<vector<double> G = generate_random_g(n, k, m);
    double result = solve(k, n, G, m);
    cout << result << "\n";
}
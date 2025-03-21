#include "data_generation.h"
//n ∈ {9, 10}, k ∈ {4, 5, 6}, m ∈ {2, 3, · · · , n − k}
using std::cout;
int main(){
    for(int i = 0; i < 5; i++){
      int n = rand() % 101;
      int k = rand() % 3 + 4;
      int m = rand() % (n - k - 2 + 1) + 2;
      vector<vector<double>> G = generate_random_g(n, k, m);
      double result = solve(k, n, G, m);
    }
}

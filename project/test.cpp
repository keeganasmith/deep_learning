#include <data_generation.h>
using std::vector, std::cout;
int main(){
    /*vector<double> row1{1, 0, .4759809, .9938236, .819425};
    vector<double> row2{0, 1, -0.8960798, -0.7442706, 0.3345122};
    vector<vector<double>> G{row1, row2};
    double result = solve(2, 5, G, 2);
    cout << "got here\n";
    cout << result << "\n";
    */
 	std::vector<std::vector<double>> matrix = {
    {1, 0, 0, 0, 0.4759809,  0.9938236, 0.819425, 1, 2},
    {0, 1, 0, 0, -0.8960798, -0.7442706, 0.3345122, 1, 2},
    {0, 0, 1, 0, 1, 2, 3, 4, 5},
    {0, 0, 0, 1, 9, 8, 7, 6, 100}
  };
  double result = solve(4, 9, matrix, 2);
	cout << result << "\n"; 
}

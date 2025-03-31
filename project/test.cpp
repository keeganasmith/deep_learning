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
    {1.0, 0.0, 0.0,  1.157,  0.017,  1.164,  1.353,  1.353},
    {0.0, 1.0, 0.0, -0.076, -0.741, -0.888, -1.495, -1.495},
    {0.0, 0.0, 1.0,  1.075, -0.702,  0.876, -0.266, -0.266}
  };
  double result = solve(3, 8, matrix, 3);
	cout << result << "\n"; 
}

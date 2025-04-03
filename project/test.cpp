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
    {1.0, 0.0, 0.0,  -.298,  -.954, 1.01 ,  -.104,  -.145, 893},
    {0.0, 1.0, 0.0, .004, .784, -1.751, -.882, -.802, -1.202},
    {0.0, 0.0, 1.0,  -.911, .346, .841, .655, .613, -.594}
  };
  double result = solve(3, 8, matrix, 3);
	cout << result << "\n"; 
}

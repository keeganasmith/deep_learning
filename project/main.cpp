#include "data_generation.h"
#include "mpi.h"
//n ∈ {9, 10}, k ∈ {4, 5, 6}, m ∈ {2, 3, · · · , n − k}
using std::cout;
struct Result{
    int n;
    int k;
    int m;
    vector<vector<double>> G;
    double result;
    Result(int n, int k, int m, vector<vector<double>>& G, double result){
        this->n = n;
        this->k = k;
        this->m = m;
        this->G = G;
        this->result = result;
    }
    string to_string() const {
        ostringstream oss;
        oss << "n: " << n << ", k: " << k << ", m: " << m << "\n";
        oss << "G (" << G.size() << " x " << (G.empty() ? 0 : G[0].size()) << "):\n";
        for (const auto &row : G) {
            for (double val : row)
                oss << val << " ";
            oss << "\n";
        }
        oss << "Result: " << result << "\n";
        oss << "------------------------------\n";
        return oss.str();
    }
};
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    vector<Result> local_results;
    for (int i = 0; i < 5; i++){
        int n = rand() % 2 + 9;            // n in {9, 10}
        int k = rand() % 3 + 4;            // k in {4, 5, 6}
        int m = rand() % (n - k - 2 + 1) + 2; // m in [2, n-k]
        vector<vector<double>> G = generate_random_g(n, k, m);
        double res = solve(k, n, G, m);
        local_results.push_back(Result{n, k, m, G, res});
    }
    ostringstream localOSS;
    for (const auto &r : local_results)
        localOSS << r.to_string();
    string localStr = localOSS.str();
    
    int localSize = localStr.size();
    
    int local_offset = 0;
    MPI_Exscan(&localSize, &local_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        local_offset = 0; // rank 0's offset is 0.
    
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, "results.txt",
                  MPI_MODE_CREATE | MPI_MODE_RDWR,
                  MPI_INFO_NULL, &fh);
    
    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);
    
    MPI_Offset my_offset = file_size + local_offset;
    
    MPI_File_write_at(fh, my_offset, localStr.c_str(), localSize, MPI_CHAR, MPI_STATUS_IGNORE);
    
    MPI_File_close(&fh);
    MPI_Finalize();
    
    return 0; 
}

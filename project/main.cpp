#include <mpi.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstring> // for memcpy
#include "data_generation.h" // Provides generate_random_g() and solve()

using namespace std;

// A simple structure to store our result.
struct Result {
    int n, k, m;
    vector<vector<double>> G;
    double result;

    // Serialize this Result to a binary buffer.
    // Format: n, k, m (ints), result (double), then k*n doubles (matrix G, row-major)
    vector<char> serialize() const {
        size_t numMatrixElements = G.size() * (G.empty() ? 0 : G[0].size());
        size_t sizeBytes = 3 * sizeof(int) + sizeof(double) + numMatrixElements * sizeof(double);
        vector<char> buffer(sizeBytes);
        char* ptr = buffer.data();
        
        // Write n, k, m.
        memcpy(ptr, &n, sizeof(int));
        ptr += sizeof(int);
        memcpy(ptr, &k, sizeof(int));
        ptr += sizeof(int);
        memcpy(ptr, &m, sizeof(int));
        ptr += sizeof(int);
        
        // Write result.
        memcpy(ptr, &result, sizeof(double));
        ptr += sizeof(double);
        
        // Write matrix data: k rows and n columns (row-major order).
        for (size_t i = 0; i < G.size(); i++) {
            for (size_t j = 0; j < G[i].size(); j++) {
                memcpy(ptr, &G[i][j], sizeof(double));
                ptr += sizeof(double);
            }
        }
        return buffer;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Each node generates 5 results.
    vector<Result> local_results;
    for (int i = 0; i < 5; i++){
        int n = rand() % 2 + 9;            // n in {9, 10}
        int k = rand() % 3 + 4;              // k in {4, 5, 6}
        int m = rand() % (n - k - 2 + 1) + 2;  // m in [2, n-k]
        vector<vector<double>> G = generate_random_g(n, k, m);
        double res = solve(k, n, G, m);
        local_results.push_back(Result{n, k, m, G, res});
    }
    
    // Serialize local results into a binary buffer.
    vector<char> localBuffer;
    for (const auto &r : local_results) {
        vector<char> ser = r.serialize();
        localBuffer.insert(localBuffer.end(), ser.begin(), ser.end());
    }
    int localSize = localBuffer.size();
    
    // Compute each process's offset using MPI_Exscan.
    int local_offset = 0;
    MPI_Exscan(&localSize, &local_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        local_offset = 0; // rank 0 starts at 0.
    
    // Open the binary file in read/write mode without truncating existing content.
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, "results.bin", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    
    // Get the current size of the file (if any data exists already).
    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);
    
    // Compute the overall offset for this process: pre-existing file size + offset from lower ranks.
    MPI_Offset my_offset = file_size + local_offset;
    
    // Write the binary buffer at the computed offset.
    MPI_File_write_at(fh, my_offset, localBuffer.data(), localSize, MPI_CHAR, MPI_STATUS_IGNORE);
    
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}

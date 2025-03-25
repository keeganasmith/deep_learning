#include <data_generation.h>
vector<vector<double>> generate_random_g(int n, int k, int m){
    int num_rows = k;
    int num_cols = n;
    vector<double> row(num_cols, 0.0);
    vector<vector<double>> result(num_rows, row);
    for(int i = 0; i < k; i++){
        result[i][i] = 1.0;
    }   
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-100.0, 100.0);
    for (int i = 0; i < num_rows; i++) {
        for (int j = k; j < num_cols; j++) {
            double x = dis(gen);
            while (fabs(x) < 1e-6) {
                x = dis(gen);
            }
            result[i][j] = x;
        }
    }

    
    for (int j = 0; j < num_cols; j++) {
        bool all_zero = true;
        for (int i = 0; i < num_rows; i++) {
            if (fabs(result[i][j]) > 1e-6) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            double x = dis(gen);
            while (fabs(x) < 1e-6) {
                x = dis(gen);
            }
            result[0][j] = x;
        }
    }

    return result;
}
// Helper: generate all combinations (subsets) of a vector 'arr' of size r.
void combinationsHelper(const vector<int>& arr, int r, int start, vector<int>& current, vector<vector<int>>& result) {
    if (current.size() == r) {
        result.push_back(current);
        return;
    }
    for (int i = start; i < arr.size(); i++) {
        current.push_back(arr[i]);
        combinationsHelper(arr, r, i + 1, current, result);
        current.pop_back();
    }
}

vector<vector<int>> combinations(const vector<int>& arr, int r) {
    vector<vector<int>> result;
    vector<int> current;
    combinationsHelper(arr, r, 0, current, result);
    return result;
}

// Helper: generate all psi vectors of length m (each element is either -1 or 1).
vector<vector<int>> generatePsi(int m) {
    vector<vector<int>> result;
    int total = 1 << m;  // 2^m possibilities
    for (int mask = 0; mask < total; mask++) {
        vector<int> psi(m);
        for (int i = 0; i < m; i++) {
            // For example, if the bit is set then assign 1, else -1.
            psi[i] = (mask & (1 << i)) ? 1 : -1;
        }
        result.push_back(psi);
    }
    return result;
}

// Solve one LP instance for a given tuple (a, b, X, psi, Y).
// G is a k x n generator matrix.
// k: number of rows.
// a, b: special column indices; X: vector of m-1 column indices; 
// psi: vector of length m; Y: remaining column indices.
double solveLPForTuple(const vector<vector<double>>& G,
                       int k, int a, int b,
                       const vector<int>& X,
                       const vector<int>& psi,
                       const vector<int>& Y) {
    glp_prob* lp = glp_create_prob();
    glp_term_out(0);
    glp_simplex(lp, NULL);
    glp_set_prob_name(lp, "m_height");
    glp_set_obj_dir(lp, GLP_MAX);  // We are maximizing

    // Add k variables (u_0,...,u_{k-1}); no bounds (free variables)
    glp_add_cols(lp, k);
    for (int i = 1; i <= k; i++) {
        glp_set_col_bnds(lp, i, GLP_FR, 0.0, 0.0);
    }

    // Count constraints:
    // For each j in X: 2 constraints,
    // one equality constraint for column b,
    // For each j in Y: 2 constraints.
    int numConstraints = 2 * X.size() + 1 + 2 * Y.size();
    glp_add_rows(lp, numConstraints);
    
    int row = 1;
    // For each j in X (first set):
    //     sum_{i=0}^{k-1} [ psi[?]*G[i][j] - psi[0]*G[i][a] ] * u_i <= 0.
    // (Here we assume that for each j in X, the corresponding sign is psi[idx+1].)
    for (size_t idx = 0; idx < X.size(); idx++) {
        glp_set_row_bnds(lp, row, GLP_UP, 0.0, 0.0);
        row++;
    }
    // For each j in X (second set):
    //     sum_{i=0}^{k-1} [ - psi[idx+1]*G[i][j] ] * u_i <= -1.
    for (size_t idx = 0; idx < X.size(); idx++) {
        glp_set_row_bnds(lp, row, GLP_UP, 0.0, -1.0);
        row++;
    }
    // Equality constraint for column b:
    glp_set_row_bnds(lp, row, GLP_FX, 1.0, 1.0);
    int eqRow = row;
    row++;
    // For each j in Y, add two constraints:
    //      sum_{i} G[i][j]*u_i <= 1  and  sum_{i} -G[i][j]*u_i <= 1.
    for (size_t idx = 0; idx < Y.size(); idx++) {
        glp_set_row_bnds(lp, row, GLP_UP, 0.0, 1.0);
        row++;
        glp_set_row_bnds(lp, row, GLP_UP, 0.0, 1.0);
        row++;
    }
    
    // Set the objective function:
    // maximize: sum_{i=0}^{k-1} [ psi[0] * G[i][a] ] * u_i.
    for (int i = 1; i <= k; i++) {
        glp_set_obj_coef(lp, i, psi[0] * G[i-1][a]);
    }

    // GLPK expects the constraint matrix in 1-indexed arrays.
    int totalCoeffs = numConstraints * k;
    vector<int> ia(totalCoeffs + 1);
    vector<int> ja(totalCoeffs + 1);
    vector<double> ar(totalCoeffs + 1);
    int pos = 1;
    
    // Fill coefficients for the constraints for each j in X (first set).
    for (size_t idx = 0; idx < X.size(); idx++) {
        int j = X[idx];
        for (int i = 1; i <= k; i++) {
            ia[pos] = idx + 1;  // row index in first set.
            ja[pos] = i;
            ar[pos] = psi[idx+1] * G[i-1][j] - psi[0] * G[i-1][a];
            pos++;
        }
    }
    // Fill coefficients for the constraints for each j in X (second set).
    for (size_t idx = 0; idx < X.size(); idx++) {
        int j = X[idx];
        for (int i = 1; i <= k; i++) {
            ia[pos] = X.size() + idx + 1;
            ja[pos] = i;
            ar[pos] = -psi[idx+1] * G[i-1][j];
            pos++;
        }
    }
    // Fill coefficients for the equality constraint for column b.
    for (int i = 1; i <= k; i++) {
        ia[pos] = eqRow;
        ja[pos] = i;
        ar[pos] = G[i-1][b];
        pos++;
    }
    // Fill coefficients for constraints for each j in Y.
    for (size_t idx = 0; idx < Y.size(); idx++) {
        int j = Y[idx];
        int currentRow = eqRow + 2 * idx + 1;
        // First constraint for column j: sum_i G[i][j]*u_i <= 1.
        for (int i = 1; i <= k; i++) {
            ia[pos] = currentRow;
            ja[pos] = i;
            ar[pos] = G[i-1][j];
            pos++;
        }
        // Second constraint for column j: sum_i -G[i][j]*u_i <= 1.
        currentRow++;
        for (int i = 1; i <= k; i++) {
            ia[pos] = currentRow;
            ja[pos] = i;
            ar[pos] = -G[i-1][j];
            pos++;
        }
    }
    
    glp_load_matrix(lp, pos - 1, ia.data(), ja.data(), ar.data());
    glp_simplex(lp, NULL);
    double obj_value = glp_get_obj_val(lp);
    glp_delete_prob(lp);
    return obj_value;
}

// Main "solve" function that takes only k, n, G, m.
double solve(int k, int n, const vector<vector<double>>& G, int m) {
    double best = -numeric_limits<double>::infinity();
    
    // Generate all psi vectors of length m.
    vector<vector<int>> allPsi = generatePsi(m);
    
    // Enumerate over tuples (a, b, X, psi).
    // a in [0, n), b in [0, n) with b != a.
    for (int a = 0; a < n; a++) {
        for (int b = 0; b < n; b++) {
            if (b == a) continue;
            // Build S = {j in [0,n) such that j != a and j != b}.
            vector<int> S;
            for (int j = 0; j < n; j++) {
                if (j == a || j == b) continue;
                S.push_back(j);
            }
            // Choose X ⊆ S with |X| = m - 1.
            vector<vector<int>> X_sets = combinations(S, m - 1);
            for (const auto& X : X_sets) {
                // Define Y = S \ X.
                vector<int> Y;
                for (int j : S) {
                    if (find(X.begin(), X.end(), j) == X.end())
                        Y.push_back(j);
                }
                // For each psi vector.
                for (const auto& psi : allPsi) {
                    double lp_val = solveLPForTuple(G, k, a, b, X, psi, Y);
                    best = max(best, lp_val);
                }
            }
        }
    }
    return best;
}

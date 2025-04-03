import struct
import numpy as np
import pandas as pd
import joblib
def read_results(file_path):
    results = []
    header_format = "<iii d"  # 3 ints and 1 double (little-endian)
    header_size = struct.calcsize(header_format)  # 20 bytes

    with open(file_path, "rb") as f:
        while True:
            header_bytes = f.read(header_size)
            if len(header_bytes) < header_size:
                break

            n, k, m, result = struct.unpack(header_format, header_bytes)
            num_matrix_elements = k * n
            matrix_bytes = f.read(num_matrix_elements * 8)
            if len(matrix_bytes) < num_matrix_elements * 8:
                print("Unexpected end of file when reading matrix data.")
                break

            matrix_flat = struct.unpack(f"{num_matrix_elements}d", matrix_bytes)
            full_G = np.array(matrix_flat, dtype=np.float64)

            # Drop first k columns from each row
            reduced_G = []
            for i in range(k):
                row_start = i * n + k
                row_end = i * n + n
                reduced_G.extend(full_G[row_start:row_end])
            G = np.array(reduced_G, dtype=np.float64)  # 1D flat

            results.append({
                "n": n,
                "k": k,
                "m": m,
                "result": result,
                "P": G  # flattened, and first k columns removed
            })

    return results

def main():
    file_path = "results.bin"
    results = read_results(file_path)
    
    # Create a pandas DataFrame from the results.
    # Note: The "G" column contains NumPy arrays, which will be stored as objects.
    df = pd.DataFrame(results)
    
    # Save the DataFrame to disk using joblib.
    joblib.dump(df, "large_results_dataframe.pkl")
    print("DataFrame saved to results_dataframe.pkl")
    
    # Display DataFrame statistics:
    print("\nDataFrame Info:")
    print(df.info())
    
    print("\nDataFrame Describe (for numeric columns):")
    print(df.describe())
    
    # Optionally, display the first few rows.
    print("\nDataFrame Head:")
    print(df.head())

if __name__ == "__main__":
    main()

import struct
import numpy as np
import pandas as pd
import joblib

def read_results(file_path):
    results = []
    header_format = "<iii d"  # 3 ints and 1 double (little-endian)
    header_size = struct.calcsize(header_format)  # should be 20 bytes
    with open(file_path, "rb") as f:
        while True:
            header_bytes = f.read(header_size)
            if len(header_bytes) < header_size:
                # End of file reached
                break
            # Unpack header: n, k, m, and computed result
            n, k, m, result = struct.unpack(header_format, header_bytes)
            # Compute how many doubles to read for the generator matrix (G)
            num_matrix_elements = k * n
            matrix_bytes = f.read(num_matrix_elements * 8)  # each double is 8 bytes
            if len(matrix_bytes) < num_matrix_elements * 8:
                print("Unexpected end of file when reading matrix data.")
                break
            # Unpack the matrix data and reshape into a k x n NumPy array
            matrix_flat = struct.unpack(f"{num_matrix_elements}d", matrix_bytes)
            G = np.array(matrix_flat).reshape((k, n))
            
            results.append({
                "n": n,
                "k": k,
                "m": m,
                "result": result,
                "G": G  # storing as a NumPy array; DataFrame will hold it as an object
            })
    return results

def main():
    file_path = "results.bin"
    results = read_results(file_path)
    
    # Create a pandas DataFrame from the results.
    # Note: The "G" column contains NumPy arrays, which will be stored as objects.
    df = pd.DataFrame(results)
    
    # Save the DataFrame to disk using joblib.
    joblib.dump(df, "results_dataframe.pkl")
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

import pandas as pd
from sklearn.cluster import KMeans
import argparse
import time

def standard_k_mean(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Detect feature columns automatically (e.g., x1, x2, ..., xN)
    feature_cols = [col for col in df.columns if col.startswith('x')]
    num_dims = len(feature_cols)

    if num_dims == 0:
        raise ValueError("No valid feature columns found (expected columns starting with 'x').")

    # Store results
    results = []

    print(f"Detected {num_dims} dimensions. Running KMeans for powers of 2 from K=2 to K=1024...")

    k = 2
    while k <= 1024:
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(df[feature_cols])
        end_time = time.time()

        exec_time = end_time - start_time
        n_iter = kmeans.n_iter_

        results.append([k, round(exec_time, 6), n_iter, num_dims])

        print(f"K={k}, Time={exec_time:.4f}s, Iter={n_iter}")

        k *= 2  # Next power of 2

    # Write to output CSV
    result_df = pd.DataFrame(results, columns=['K', 'ExecTime', 'ConvergedIn', 'Dim'])
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KMeans clustering for powers of 2.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file for results')
    args = parser.parse_args()

    standard_k_mean(args.input, args.output)

import cudf
import numpy as np
import pandas as pd  # For saving results
from cuml.cluster import KMeans
import time

def run_kmeans_on_csv(filename, output_csv="kmeans_results.csv"):
    df = cudf.read_csv(filename)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} features.")

    results = []

    for k in [2**i for i in range(1, 11)]:
        model = KMeans(n_clusters=k, init="k-means++", max_iter=300)

        start = time.time()
        model.fit(df)
        end = time.time()

        inertia = model.inertia_
        duration = end - start

        print(f"K={k:4}, Time={duration:.4f}s, Inertia={inertia:.2f}")

        results.append({
            "K": k,
            "Time(s)": round(duration, 6),
            "Inertia": round(inertia, 6)
        })

    # Save results to CSV using pandas
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gpu_kmeans_cuda_ml.py <data.csv> [output.csv]")
        exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "kmeans_results.csv"
    run_kmeans_on_csv(input_file, output_file)

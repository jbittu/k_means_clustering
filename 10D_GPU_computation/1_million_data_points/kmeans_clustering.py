import pandas as pd
from sklearn.cluster import KMeans
import argparse
import time

def perform_kmeans(input_file, output_file, n_clusters):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Check if features x1 to x10 exist
    features = [f'x{i}' for i in range(1, 11)]
    if not all(f in df.columns for f in features):
        raise ValueError("Input file must contain columns x1 through x10.")

    # Start timing the clustering process
    start_time = time.time()

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(df[features])

    # End timing
    end_time = time.time()
    duration = end_time - start_time
    print(f"Clustering complete in {duration:.2f} seconds.")

    # Save to new file
    df.to_csv(output_file, index=False)
    print(f"Output written to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='K-Means clustering on a 10D dataset.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--clusters', type=int, required=True, help='Number of clusters (K)')
    args = parser.parse_args()

    perform_kmeans(args.input, args.output, args.clusters)

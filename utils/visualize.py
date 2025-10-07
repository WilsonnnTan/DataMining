from utils.preprocess import load_csv, check_missing_value, remove_missing_rows, check_duplicates, remove_duplicates, encode_kategori, data_reduction, find_non_informative_columns
from utils.kmodes import kmodes, kmodes_cost


def cluster_data(file_path):
    # Load data
    header, rows = load_csv(file_path)

    # Check missing values
    if check_missing_value(rows):
        rows = remove_missing_rows(rows)
    
    # Check duplicates
    if check_duplicates(rows):
        rows = remove_duplicates(rows)
    
    # Preprocess
    header, rows = encode_kategori(header, rows)

    # K-Modes clustering
    clusters, centroids = kmodes(rows=rows)
    cost = kmodes_cost(clusters, centroids)
    print(f"K-Modes cost: {cost}")
    
    # Analyze columns
    non_info_cols = find_non_informative_columns(centroids, header)
    data_reduction(non_info_cols, header, rows)
    
    # K-Modes clustering
    clusters, centroids = kmodes(rows=rows)
    cost = kmodes_cost(clusters, centroids)
    print(f"K-Modes cost: {cost}")
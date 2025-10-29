import numpy as np

# ------------------- Step 1: Generate sample data (Version 2) -------------------
def generate_sample_data_v2():
    rows = int(input("Number of rows: "))
    cols = int(input("Number of columns: "))

    data = np.zeros((rows, cols))

    for i in range(cols):
        print(f"Column {i + 1}:")
        dist = input("  Distribution (normal/uniform): ").strip().lower()
        mean = float(input("Mean: "))

        if dist == "normal":
            std = float(input("Standard Deviation: "))
            data[:, i] = np.random.normal(loc=mean, scale=std, size=rows)
        elif dist == "uniform":
            range_val = float(input("Range value: "))
            data[:, i] = np.random.uniform(low=mean - range_val, high=mean + range_val, size=rows)
        else:
            print("Invalid distribution (default normal distribution)")
            data[:, i] = np.random.normal(loc=mean, scale=1.0, size=rows)

    print(data)
    return data

# ------------------- Step 2: Mean centering -------------------
def prepare_data_for_pca(data, scale=False):
    data_centered = data - np.mean(data, axis=0)
    if scale:
        std_dev = np.std(data_centered, axis=0)
        std_dev[std_dev == 0] = 1
        data_centered /= std_dev
    return data_centered

# ------------------- Step 3: Covariance matrix-------------------
def covariance_matrix(data):
    data_centered = data - np.mean(data, axis=0)

    n = data.shape[0]

    #Covariance matrix formula: (X^T X) / (n - 1)
    cov_matrix = (data_centered.T @ data_centered) / (n - 1)

    print(cov_matrix)
    return cov_matrix

if __name__ == "__main__":
    data = generate_sample_data_v2()
    covariance_matrix(data)

# ------------------- Step 3: Eigenvalues and Eigenvectors -------------------

def eigen(cov_matrix):
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    
    if not np.allclose(cov_matrix, cov_matrix.T):
        raise ValueError("Matrix must be symmetric.")

    if np.any(cov_matrix < 0):
        raise ValueError("Matrix must have all nonnegative entries.")

    char_poly = np.poly(cov_matrix)
    print("\nCharacteristic polynomial coefficients:")
    print(char_poly)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]  
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print("Eigenvalues:")
    print(eigenvalues)
    print("Eigenvectors:")
    print(eigenvectors)

    return eigenvalues, eigenvectors
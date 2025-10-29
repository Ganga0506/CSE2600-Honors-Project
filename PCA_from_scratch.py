import numpy as np

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
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

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def read_estimated_times(filename):
    with open(filename, "r") as file:
        times = [float(line.strip()) for line in file]
    return times


def plot_estimated_times(filename=None, data=None):
    if filename:
        estimated_times = read_estimated_times(filename)
    elif data:
        estimated_times = data

    x = list(range(1, len(estimated_times) + 1))

    # Plotting
    plt.figure(figsize=(10, 6))  # Adjusts the figure size
    plt.plot(
        x, estimated_times, marker="o"
    )  # 'o' creates circular markers for each data point
    plt.title("Estimated Time Left")  # Title of the plot
    plt.xlabel("Measurement Number")  # X-axis label
    plt.ylabel("Time Left (seconds)")  # Y-axis label
    plt.grid(True)  # Adds a grid for easier reading
    plt.show()


def plot_execution_times_prob():
    file_path = "diffs.txt"
    with open(file_path, "r") as file:
        execution_times = [float(line.strip()) for line in file]

    data = np.array(execution_times)

    stats.probplot(data, plot=plt)

    plt.title("Probability plot of Execution Times")
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Ordered Values")
    plt.grid(True)
    plt.show()


def iqr_filter(diffs):
    q1 = diffs.quantile(0.25)
    q3 = diffs.quantile(0.75)

    # Calculate IQR
    iqr = q3 - q1

    # Define bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter out outliers
    filtered_durations = diffs[(diffs >= lower_bound) & (diffs <= upper_bound)]
    return filtered_durations


def histogram():
    import pandas as pd

    # Step 1: Read execution times from the file
    file_path = "./diffs.txt"
    with open(file_path, "r") as file:
        # Convert each line to a float and store in a list
        execution_times = [float(line.strip()) for line in file]

    # Step 2: Plot the histogram
    spans = []
    w = 15
    for i in range(0, len(execution_times), w):
        span = execution_times[i : i + w]
        iqr_filtered = iqr_filter(pd.Series(span))
        spans.append(iqr_filtered)

    execution_times = pd.concat(spans)

    plt.hist(execution_times, bins="auto", alpha=0.7, rwidth=0.85)
    plt.title("Histogram of Execution Times")
    plt.xlabel("Execution Time")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    histogram()

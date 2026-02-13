import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.normal(0, 1, n)

    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal(0,1) Distribution")
    plt.show()
    
    return data


def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.uniform(0, 10, n)

    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform(0,10) Distribution")
    plt.show()
    
    return data


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.binomial(1, 0.5, n)
    
    plt.hist(data, bins=10)
    plt.xlabel("Value (0 or 1)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli(0.5) Distribution")
    plt.show()

    return data 


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    arr = np.array(data)
    
    n = len(arr)
    mean = np.sum(arr) / n
    
    return mean

def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    arr = np.array(data)
    
    n = len(arr)
    mean = sample_mean(arr)
    
    # Using (n - 1) in denominator
    variance = np.sum((arr - mean) ** 2) / (n - 1)

    return variance
# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Use a consistent quartile definition. The tests for the fixed
    dataset [5,1,3,2,4] expect Q1=2 and Q3=4.
    """
    arr = np.array(data)
    arr = np.sort(arr)
    
    n = len(arr)
    
    minimum = arr[0]
    maximum = arr[-1]
    
    # Median
    if n % 2 == 0:
        median = (arr[n//2 - 1] + arr[n//2]) / 2
        lower_half = arr[:n//2]
        upper_half = arr[n//2:]
    else:
        median = arr[n//2]
        lower_half = arr[:n//2 + 1]
        upper_half = arr[n//2:]
    
    # Q1
    m = len(lower_half)
    if m % 2 == 0:
        q1 = (lower_half[m//2 - 1] + lower_half[m//2]) / 2
    else:
        q1 = lower_half[m//2]
    
    # Q3
    m = len(upper_half)
    if m % 2 == 0:
        q3 = (upper_half[m//2 - 1] + upper_half[m//2]) / 2
    else:
        q3 = upper_half[m//2]
    
    return (minimum, maximum, median, q1, q3)

# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)
    
    mean_x = np.sum(x) / n
    mean_y = np.sum(y) / n
    
    covariance = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    
    return covariance


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    x = np.array(x)
    y = np.array(y)
    
    # Variances (sample variance with n-1)
    var_x = sample_covariance(x, x)
    var_y = sample_covariance(y, y)
    
    # Covariance
    cov_xy = sample_covariance(x, y)
    
    # Construct 2x2 matrix
    matrix = np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])
    
    return matrix

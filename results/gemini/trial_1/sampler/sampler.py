import sys
import numpy as np
import pandas as pd

def generate_data(n_rows: int, seed: int) -> pd.DataFrame:
    """
    Generates a pandas DataFrame with specified columns and distributions.

    Args:
        n_rows: The number of rows to generate.
        seed: The seed for the random number generator.

    Returns:
        A pandas DataFrame with columns: normal, lognormal, bimodal, x, y, category.
    """
    rng = np.random.default_rng(seed)

    # 1. normal ~ Normal(mean=0, standard deviation=1)
    normal_data = rng.normal(loc=0, scale=1, size=n_rows)

    # 2. lognormal = exp(Z), Z ~ Normal(0,0.75)
    z_data = rng.normal(loc=0, scale=0.75, size=n_rows)
    lognormal_data = np.exp(z_data)

    # 3. bimodal: with probability 0.4 Normal(-2,0.5), otherwise Normal(2,0.8)
    # Generate a mask to choose between the two distributions
    choice_mask = rng.random(n_rows) < 0.4
    
    # Generate data for both parts of the bimodal distribution
    bimodal_part1 = rng.normal(loc=-2, scale=0.5, size=n_rows)
    bimodal_part2 = rng.normal(loc=2, scale=0.8, size=n_rows)
    
    # Combine them based on the choice mask
    bimodal_data = np.where(choice_mask, bimodal_part1, bimodal_part2)

    # 4. x,y jointly standard normal, correlation 0.8
    # Mean vector for standard normal is [0, 0]
    mean = [0, 0]
    
    # Covariance matrix:
    # C = [[var_x, cov_xy],
    #      [cov_yx, var_y]]
    # For standard normal, var_x = var_y = 1.
    # cov_xy = correlation * std_x * std_y = 0.8 * 1 * 1 = 0.8.
    covariance_matrix = [[1, 0.8],
                         [0.8, 1]]
    
    xy_data = rng.multivariate_normal(mean=mean, cov=covariance_matrix, size=n_rows)
    x_data = xy_data[:, 0]
    y_data = xy_data[:, 1]

    # 5. category takes A,B,C with probabilities 0.5,0.3,0.2
    categories = ['A', 'B', 'C']
    probabilities = [0.5, 0.3, 0.2]
    category_data = rng.choice(categories, size=n_rows, p=probabilities)

    # Create a dictionary of data to form the DataFrame
    data_dict = {
        'normal': normal_data,
        'lognormal': lognormal_data,
        'bimodal': bimodal_data,
        'x': x_data,
        'y': y_data,
        'category': category_data
    }
    
    # Create DataFrame with specified column order
    df = pd.DataFrame(data_dict)

    return df

if __name__ == "__main__":
    # Check for correct number of command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <n_rows> <seed>", file=sys.stderr)
        sys.exit(1)

    try:
        # Parse n_rows and seed from command-line arguments
        n_rows = int(sys.argv[1])
        seed = int(sys.argv[2])
    except ValueError:
        print("Error: n_rows and seed must be integers.", file=sys.stderr)
        sys.exit(1)

    # Generate the DataFrame
    df = generate_data(n_rows, seed)

    # Write the DataFrame to standard output as a CSV, without the index
    df.to_csv(sys.stdout, index=False)
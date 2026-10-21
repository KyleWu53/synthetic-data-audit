import sys
import numpy as np
import pandas as pd

def generate_data(n: int, seed: int) -> pd.DataFrame:
    """
    Generates a pandas DataFrame with specified columns and distributions.

    Args:
        n (int): The number of rows to generate.
        seed (int): The seed for the random number generator.

    Returns:
        pd.DataFrame: A DataFrame containing the generated data.
    """
    rng = np.random.default_rng(seed)

    # 1. normal ~ Normal(mean=0, standard deviation=1)
    normal_data = rng.normal(loc=0, scale=1, size=n)

    # 2. lognormal = exp(Z), Z ~ Normal(0,0.75)
    z_data = rng.normal(loc=0, scale=0.75, size=n)
    lognormal_data = np.exp(z_data)

    # 3. bimodal: with probability 0.4 Normal(-2,0.5), otherwise Normal(2,0.8)
    choice_mask = rng.choice([True, False], p=[0.4, 0.6], size=n)
    bimodal_data_neg = rng.normal(loc=-2, scale=0.5, size=n)
    bimodal_data_pos = rng.normal(loc=2, scale=0.8, size=n)
    bimodal_data = np.where(choice_mask, bimodal_data_neg, bimodal_data_pos)

    # 4. x,y jointly standard normal, correlation 0.8
    # Mean vector for x,y
    mean_xy = [0, 0]
    # Covariance matrix: [[var_x, cov_xy], [cov_yx, var_y]]
    # For standard normal, var_x = var_y = 1.
    # cov_xy = correlation * std_x * std_y = 0.8 * 1 * 1 = 0.8
    cov_xy = [[1, 0.8], [0.8, 1]]
    xy_data = rng.multivariate_normal(mean=mean_xy, cov=cov_xy, size=n)
    x_data = xy_data[:, 0]
    y_data = xy_data[:, 1]

    # 5. category takes A,B,C with probabilities 0.5,0.3,0.2
    category_data = rng.choice(['A', 'B', 'C'], p=[0.5, 0.3, 0.2], size=n)

    # Create a dictionary of the generated data
    data = {
        'normal': normal_data,
        'lognormal': lognormal_data,
        'bimodal': bimodal_data,
        'x': x_data,
        'y': y_data,
        'category': category_data
    }

    # Define the exact column order as required
    column_order = ['normal', 'lognormal', 'bimodal', 'x', 'y', 'category']

    # Create the DataFrame with the specified column order
    df = pd.DataFrame(data, columns=column_order)

    return df

if __name__ == '__main__':
    # Read n and seed from command line arguments
    # sys.argv[0] is the script name, sys.argv[1] is n, sys.argv[2] is seed
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <n> <seed>", file=sys.stderr)
        sys.exit(1)

    try:
        n_rows = int(sys.argv[1])
        random_seed = int(sys.argv[2])
    except ValueError:
        print("Error: n and seed must be integers.", file=sys.stderr)
        sys.exit(1)

    # Generate the DataFrame
    result_df = generate_data(n_rows, random_seed)

    # Write the DataFrame to stdout in CSV format, without the index
    result_df.to_csv(sys.stdout, index=False)
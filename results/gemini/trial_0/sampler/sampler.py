import sys
import numpy as np
import pandas as pd

def generate_data():
    """
    Generates a pandas DataFrame with specified columns and distributions.
    Reads n (number of rows) and seed from command line arguments.
    Writes the DataFrame as CSV to standard output.
    """

    # 1. Read n and seed from sys.argv
    if len(sys.argv) != 3:
        print("Usage: python script.py <n_rows> <seed>", file=sys.stderr)
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        seed = int(sys.argv[2])
    except ValueError:
        print("Error: n_rows and seed must be integers.", file=sys.stderr)
        sys.exit(1)

    # 2. Initialize the random number generator
    rng = np.random.default_rng(seed)

    # 3. Generate data for each column according to specifications

    # normal ~ Normal(mean=0, standard deviation=1)
    normal_data = rng.normal(loc=0, scale=1, size=n)

    # lognormal = exp(Z), Z ~ Normal(0,0.75)
    z_data = rng.normal(loc=0, scale=0.75, size=n)
    lognormal_data = np.exp(z_data)

    # bimodal: with probability 0.4 Normal(-2,0.5), otherwise Normal(2,0.8)
    bimodal_data = np.zeros(n)
    # Create a boolean mask for the first mode
    choice_mask_mode1 = rng.uniform(size=n) < 0.4
    
    # Generate data for mode 1 (Normal(-2,0.5)) where mask is True
    num_mode1_samples = np.sum(choice_mask_mode1)
    bimodal_data[choice_mask_mode1] = rng.normal(loc=-2, scale=0.5, size=num_mode1_samples)
    
    # Generate data for mode 2 (Normal(2,0.8)) where mask is False
    num_mode2_samples = n - num_mode1_samples
    bimodal_data[~choice_mask_mode1] = rng.normal(loc=2, scale=0.8, size=num_mode2_samples)

    # x,y jointly standard normal, correlation 0.8
    mean_xy = [0, 0]
    # Covariance matrix for standard normal variables with correlation 0.8
    # cov(X,Y) = correlation * std(X) * std(Y) = 0.8 * 1 * 1 = 0.8
    cov_xy = [[1, 0.8], 
              [0.8, 1]]
    xy_joint_data = rng.multivariate_normal(mean=mean_xy, cov=cov_xy, size=n)
    x_data = xy_joint_data[:, 0]
    y_data = xy_joint_data[:, 1]

    # category takes A,B,C with probabilities 0.5,0.3,0.2
    category_data = rng.choice(['A', 'B', 'C'], size=n, p=[0.5, 0.3, 0.2])

    # 4. Create a pandas DataFrame
    data_dict = {
        'normal': normal_data,
        'lognormal': lognormal_data,
        'bimodal': bimodal_data,
        'x': x_data,
        'y': y_data,
        'category': category_data
    }
    df = pd.DataFrame(data_dict)

    # 5. Ensure the specified column order
    column_order = ['normal', 'lognormal', 'bimodal', 'x', 'y', 'category']
    df = df[column_order]

    # 6. Write the DataFrame to standard output as CSV, including header, no index
    df.to_csv(sys.stdout, index=False)

if __name__ == "__main__":
    generate_data()
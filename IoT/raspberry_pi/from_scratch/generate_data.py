


import numpy as np
import pandas as pd

# Generate synthetic data for a simple regression
np.random.seed(42)
x = np.random.rand(100, 1) * 10 # Features
y = 2 * x + 1 + np.random.randn(100, 1) * 2 # Targets with some noise

# convert to dataframe
data = pd.DataFrame(np.hstack((x,y)),  columns=['x', 'y'])
data.to_csv('dataset.csv', index=False)

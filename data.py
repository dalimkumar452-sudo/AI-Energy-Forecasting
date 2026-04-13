import pandas as pd
import numpy as np

# Create folder automatically (safe)
import os
os.makedirs('data', exist_ok=True)

# Create datetime range
date_range = pd.date_range(start='2023-01-01', periods=1000, freq='H')

# Generate realistic energy data
energy = 100 + 20*np.sin(np.arange(1000)/24 * 2*np.pi) + np.random.normal(0, 5, 1000)

# Create dataframe
df = pd.DataFrame({
    'Datetime': date_range,
    'Energy': energy
})

# Save CSV
df.to_csv('data/energy.csv', index=False)

print("✅ energy.csv created successfully!")
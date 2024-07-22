from sklearn.preprocessing import RobustScaler
import numpy as np

# Create some sample data with outliers
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])

# Reshape the data for sklearn
data = data.reshape(-1, 1)

# Initialize the RobustScaler
scaler = RobustScaler()

# Fit the scaler on the data and transform it
scaled_data = scaler.fit_transform(data)

print(scaled_data)

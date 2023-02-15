import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load CSV file
data = pd.read_csv("applerevenue .csv", index_col=0, parse_dates=True)

# Plot revenue data
data.plot()
plt.show()

# Fit ARIMA model
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()

# Make predictions
start = len(data)
end = len(data) + 10
predictions = model_fit.predict(start=start, end=end, typ='levels')

# Plot predictions
predictions.plot()
plt.show()

import pandas as pd
from sktime.forecasting.arima import AutoARIMA

# Define the directory path and file name
sim_dir = r"/data/simulated/"
file_name = "noisy_simdata.csv"

# Combine the directory path and file name
file_path = sim_dir + file_name

# Read and preprocess Dataset
df = pd.read_csv(file_path)
df["Date"] = pd.PeriodIndex(pd.to_datetime(df["Date"], format="%d-%m-%Y"), freq="M")
df.set_index("Date", inplace=True)

# Now you can work with the data DataFrame
# For example, you can print the first few rows
print(df.head())

# The following data preprocessing should be above the final wrapper function


# Convert DataFrame to TimeSeries and split target and predictors
predictors_names = [
    "x" + str(x + 1) for x in range(df.shape[1] - 1)
]  # allows that the number of predictors can be generic; ensure in data preprocessing that target is named 'y' and predictors 'x1', 'x2' etc.
df[predictors_names] = df[predictors_names].shift(1)
df.dropna(inplace=True)


# Split the data into training and validation sets
train_split = 0.3
train_size = int(df.shape[0] * train_split)
train = df[:train_size]

fc_horizon = 1
current_predictors = None

# Setting up Auto-SARIMAX Model without covariates
model = AutoARIMA(
    seasonal=True,
    sp=12,
    trace=True,
    with_intercept="auto",
    max_p=12,
    max_q=12,
    suppress_warnings=True,
)

# Train Model on Training Data
model.fit(train["y"], None)

prediction = model.predict(fc_horizon, current_predictors)

print(prediction)

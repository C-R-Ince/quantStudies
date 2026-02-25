import pandas as pd
import numpy as np
import sys
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description="Natural Gas Forecast")
parser.add_argument("-d", "--date",
                    required=True,
                    help="Input date in MM/DD/YY format")
parser.add_argument("-p", "--filepath",
                    required=True,
                    help="Path to CSV file")
args = parser.parse_args()

#Inputs
natGas = pd.read_csv(args.filepath, header = 0)
natGas["Dates"] = pd.to_datetime(natGas["Dates"], format="%m/%d/%y")
parsedDate = args.date

#Formats input
def validDate(date, format='%m/%d/%y'):
    try:
        datetime.strptime(date, format)
        return datetime.strptime(date, format)
    except ValueError:
        return False
#Gets the next year date
def nxtYear(date, format='%m/%d/%y'):
    try: 
        return date.replace(year=date.year + 1)
    except ValueError:
        return date.replace(month=2, day=28, year=date.year + 1)
#Generate yearly window
def yearlyRole(df):
    resultMax = []
    resultMin = []
    years = df['Dates'].dt.year.unique()
    for year in years:
        start = pd.Timestamp(year=year, month=4, day=28)
        end = start + pd.DateOffset(years=1)
        window = df[(df['Dates'] >= start) & (df['Dates'] <= end)]
        resultMax.append(window["Prices"].max())
        resultMin.append(window["Prices"].min())
        
    return pd.DataFrame({"year": years, "min": resultMin, "max": resultMax})
#Checks input is valid
inputDate = validDate(parsedDate)
if inputDate is None:
    print("Invalid date format. Use MM/DD/YY")
    sys.exit 

# Define axis values
X = natGas["Dates"]
y = natGas["Prices"]

# Plot
fig, ax = plt.subplots()
ax.plot(X, y, '-')
ax.set_xlabel('Dates')
ax.set_ylabel('Price')
ax.set_title('Natural Gas Prices')
ax.tick_params(axis='x', rotation=45)
plt.show()

# Convert dates to ordinal for regression
X_ord = natGas["Dates"].map(datetime.toordinal).values.reshape(-1, 1)

# Fit model
reg = LinearRegression().fit(X_ord, y)

# Generate future monthly dates
today = pd.Timestamp.today()
X_new_dates = pd.date_range(
    start=natGas["Dates"].min(),
    end=today,
    freq="ME"
)

# Convert future dates to ordinal
X_new_ord = X_new_dates.map(datetime.toordinal).values.reshape(-1, 1)

# Predict
y_new = reg.predict(X_new_ord)

futureNatGas = pd.DataFrame()
futureNatGas["Prices"] = y_new
futureNatGas["Dates"] = X_new_dates

#Plot Linear Regression data
fig, ax = plt.subplots()
ax.plot(futureNatGas["Dates"], futureNatGas["Prices"], '-')
ax.set_xlabel('Dates')
ax.set_ylabel('Price')
ax.set_title('Natural Gas Prices')
ax.tick_params(axis='x', rotation=45)
plt.show()

#Create Average amplitude
avgData = yearlyRole(natGas)
aMin = avgData["min"].mean()
aMax = avgData["max"].mean()

#Fill values for sinusoid
t = 12
t_step = np.arange(len(futureNatGas))
A = (aMax - aMin) / 2
offset = np.mean(natGas["Prices"])
phase = 0
omega = 2*np.pi/t
y_sin = A * np.sin(omega * t_step + phase) + offset

#Combine sinusoid and regression and substract prices as offset
combine = y_sin + y_new - offset

#Plot including actual prices, linear regression, fitted sinusoidal trend
plt.figure(figsize=(10,5))
plt.plot(natGas["Dates"], natGas["Prices"], label="Actual Prices")
plt.plot(futureNatGas["Dates"], futureNatGas["Prices"], label="Linear Regression", linestyle='--', color='red')
plt.plot(futureNatGas["Dates"], combine, label="Sinusoidal Trend", color="orange")
plt.legend()
plt.xticks(rotation=45)
plt.show()

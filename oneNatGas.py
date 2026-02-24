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
    result = []
    # Get unique years in the data
    years = df['Dates'].dt.year.unique()
    for year in years:
        start = pd.Timestamp(year=year, month=4)
        end = start + pd.DateOffset(years=1)
        window = df[(df['date'] >= start) & (df['date'] <= end)]
        yearMax = df["Prices"].max()
        yearMin = df["Prices"].min()
        yearDifference = yearMax - yearMin
        return yearDifference
    return 

#Checks input is valid
inputDate = validDate(parsedDate)
if inputDate is None:
    print("Invalid date format. Use MM/DD/YY")
    sys.exit 

X = natGas["Dates"]
y = natGas["Prices"]

#Plot figure
fig, ax = plt.subplots()
ax.plot(natGas["Dates"], y, '-')
ax.set_xlabel('Dates')
ax.set_ylabel('Price')
ax.set_title('Natural Gas Prices')
ax.tick_params(axis='x', rotation=45)
plt.show()

#Generate data for future months
today = pd.Timestamp.today()
X_new = pd.date_range(
    start=natGas["Dates"].min(),
    end=today,
    freq="ME"
)
X_new = pd.DataFrame(X_new, columns=["Datess"])
X_new["Dates"] = X_new[["Datess"]].map(datetime.toordinal)

#Get linear regression
X = natGas[["Dates"]].map(datetime.toordinal)
reg = LinearRegression().fit(X, y)

#Predict new values
y_new = reg.predict(X_new[["Dates"]])
print(X_new)

#Plot linear regression
fig, ax = plt.subplots()
ax.plot(X_new["Datess"], y_new, '-')
ax.set_xlabel('Dates')
ax.set_ylabel('Price')
ax.set_title('Natural Gas Prices')
ax.tick_params(axis='x', rotation=45)
plt.show()


"""
#Quieries
#rowMatch = natGas[natGas["Datess"] == inputDate]
#nxtYearRow = natGas[natGas["Datess"] == nxtYearDate]
"""

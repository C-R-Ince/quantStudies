import pandas as pd
import numpy as np
import sys
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

#Parse arguments
parser = argparse.ArgumentParser(description="Natural Gas Forecast")
parser.add_argument("-p", "--filepath",
                    required=True,
                    help="Path to CSV file")
parser.add_argument("-i", "--injectionDate",
                    required=True,
                    help="Input date in MM/DD/YY format")
parser.add_argument("-r", "--rate",
                    required=True,
                    help="Rate at which gas can be injected/withdrawn")
parser.add_argument("-v", "--volumeStorage",
                    required=True,
                    help="Int total storage capacity in millions MMBtu")
parser.add_argument("-c", "--costStorage",
                    required=True,
                    help="Int for value of storage per month in per 1 million MMBtu")
parser.add_argument("-t", "--transport",
                    required=True,
                    help="Int for cost of transport")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-w", "--withdrawalDate",
                    help="Input date in MM/DD/YY format")
group.add_argument("-m", "--holdingMonths",
                    help="Input number of months until sale")

args = parser.parse_args()

def validDate(date, format = '%m/%d/%y'):
    try:
        parsedDate = datetime.strptime(date, format)
        return datetime.strptime(date, format)
    except ValueError:
        print("Please enter dates as MM/DD/YY.")
        sys.exit(1)

    if parsedDate < minDate or parsedDate > maxDate:
        print("Valid dates are between ", minDate, " and ", maxDate)
        sys.exit(1)

    return parsedDate

injectionDate = validDate(args.injectionDate)
rate = float(args.rate)
volumeStorage = float(args.volumeStorage)
costStorage = float(args.costStorage)
transport = float(args.transport)

withdrawalDate = validDate(args.withdrawalDate) if args.withdrawalDate else None
holdingMonths = float(args.holdingMonths) if args.holdingMonths else None

#Inputs
natGas = pd.read_csv(args.filepath, header = 0)
natGas["Dates"] = pd.to_datetime(natGas["Dates"], format="%m/%d/%y")

maxDate = natGas["Dates"].max()
minDate = natGas["Dates"].min()

if withdrawalDate:
    withdrawalPrice = natGas["Prices"].loc[natGas["Dates"] == withdrawalDate].values[0]
    months = (withdrawalDate.year - injectionDate.year) * 12 + (withdrawalDate.month - injectionDate.month)
    totalStorageCost = months * costStorage
elif holdingMonths:
    withDate = injectionDate + relativedelta(months = int(holdingMonths))
    withdrawalPrice = natGas["Prices"].loc[natGas["Dates"] == withDate].values[0]
    totalStorageCost = costStorage * holdingMonths


injectionPrice = natGas["Prices"].loc[natGas["Dates"] == injectionDate].values[0]
profit = (
    (withdrawalPrice*volumeStorage) 
    - (
        (injectionPrice*volumeStorage) 
        + totalStorageCost 
        + (rate*2) 
        + (transport*2)
        )
    )

profit = profit * 1000000

print(f"Projected Profit: ${profit:,.2f}")

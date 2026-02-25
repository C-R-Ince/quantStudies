import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

#Input fin info
parser = argparse.ArgumentParser(description="Compute PD and Expected Loss for a loan")
parser.add_argument("-c", "--credit_lines_outstanding", type=int, required=True, help="Outstanding credit lines")
parser.add_argument("-l", "--loan_amt_outstanding", type=float, required=True, help="Remaining loan amount")
parser.add_argument("-d", "--total_debt_outstanding", type=float, required=True,help="Outstanding debt")
parser.add_argument("-i",  "--income", type=float, required=True,help="Income")
parser.add_argument("-e", "--years_employed", type=float, required=True, help="Total years of employment")
parser.add_argument("-f", "--fico_score", type=int, required=True, help="Fico score")

args = parser.parse_args()

#Input loan info for model
loanData = pd.read_csv("./Loan_Data.csv", header = 0)

#Create income ratios
loanData["incomeDebt"] = loanData["total_debt_outstanding"] / loanData["income"]
loanData["incomeLoan"] = loanData["loan_amt_outstanding"] / loanData["income"]

#Create fico categories 
conditions = [
    (loanData["fico_score"] >= 800) & (loanData["fico_score"] <= 850),
    (loanData["fico_score"] >= 740) & (loanData["fico_score"] <= 799),
    (loanData["fico_score"] >= 670) & (loanData["fico_score"] <= 739),
    (loanData["fico_score"] >= 660) & (loanData["fico_score"] <= 669),
    (loanData["fico_score"] <= 579)
]
choices = ["exceptional", "veryGood", "good", "fair", "poor"]
loanData["ficoBand"] = np.select(conditions, choices, default="unknown")
#Encode fico
le = LabelEncoder()
loanData["ficoBandEncode"] = le.fit_transform(loanData["ficoBand"])
#List features for random forest
features = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
    "incomeDebt",
    "incomeLoan",
    "ficoBandEncode"
]
#Split data for random forest
X = loanData[features]
y = loanData["default"]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#Random Forest Parameters
rf = RandomForestClassifier(
    n_estimators = 300,
    max_depth = None,
    min_samples_split = 100,
    class_weight = "balanced"
    )
#Train random forest
rf.fit(Xtrain, ytrain)

#Test Random Forest
yPredProb = rf.predict_proba(Xtest)[:,1]  # predicted PD
roc = roc_auc_score(ytest, yPredProb)
print(f"ROC-AUC: {roc:.3f}")
#Test against validation
scores = cross_val_score(rf, Xtrain, ytrain, cv=5, scoring='roc_auc')
print("CV ROC-AUC scores: ", scores)
print(f"Mean CV ROC-AUC:, {scores.mean():.3f}")

#Function to calculate expected loss
def expectedLoss(credit_lines_outstanding,loan_amt_outstanding,total_debt_outstanding,income,years_employed, fico_score):
    #Create income ratios
    incomeDebt = total_debt_outstanding / income
    incomeLoan = loan_amt_outstanding / income
    #Encode ficoband
    if fico_score >= 800 & fico_score <= 850:
        ficoBandEncode = 0 #Exceptional
    elif fico_score >= 740 & fico_score <= 799:
        ficoBandEncode = 1 #veryGood
    elif fico_score >= 670 & fico_score <= 739:
        ficoBandEncode = 2 #good
    elif fico_score >= 660 & fico_score <= 669:
        ficoBandCode = 3 #fair
    elif fico_score <= 579:
        ficoBandEncode = 4 #poor
    else:
        ficoBandEncode = 5 #Unknown
    #Create df from new information
    XNew = pd.DataFrame([[
        credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
        income, years_employed, fico_score, incomeDebt, incomeLoan, ficoBandEncode
    ]], columns=features)
    #
    PD = rf.predict_proba(XNew)[:,1][0]
    # Expected Loss
    LGD = 0.9
    EL = PD * LGD * loan_amt_outstanding
    return PD, EL

#User inputted loans
pdPred, elValue = expectedLoss(
    args.credit_lines_outstanding,
    args.loan_amt_outstanding,
    args.total_debt_outstanding,
    args.income,
    args.years_employed,
    args.fico_score
)

#Ouput probability and expected loss
print(f"Predicted PD: {pdPred:.3f}, Expected Loss: ${elValue:,.2f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Input loan info for model
loanData = pd.read_csv("./Loan_Data.csv", header = 0)

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

#Group by bin
grouped = loanData.groupby("ficoBand")
#Get number of defaults per bin
ki = grouped["default"].apply(lambda x: (x != 0).sum()).to_numpy()
#Get total number per bin
ni = grouped["default"].count().to_numpy()
#Estimate probability for this bin (MLE)
pi = np.clip(ki / ni, 1e-6, 1-1e-6)
#Store bin cont. 
LLbin = ki * np.log(pi) + (ni - ki) * np.log(1 - pi)

#Define probabilities for bins
p1_vals = np.linspace(0.01, 0.99, 50)
p2_vals = np.linspace(0.01, 0.99, 50)

#Compute LL
LL_surface = np.zeros((len(p1_vals), len(p2_vals)))
pi_fixed = ki[2:] / ni[2:]

for i, p1 in enumerate(p1_vals):
    for j, p2 in enumerate(p2_vals):
        p_all = np.concatenate(([p1, p2], pi_fixed))
        p_all = np.clip(p_all, 1e-6, 1-1e-6)
        LL = np.sum(ki * np.log(p_all) + (ni - ki) * np.log(1 - p_all))
        LL_surface[i, j] = LL
#Plot 
plt.figure(figsize=(8,6))
plt.contourf(p1_vals, p2_vals, LL_surface.T, levels=30, cmap='viridis')
plt.colorbar(label='Log-Likelihood')
plt.xlabel('p1 (bin 1 probability)')
plt.ylabel('p2 (bin 2 probability)')
plt.title('Log-Likelihood Surface for First Two FICO Bins')
plt.show()


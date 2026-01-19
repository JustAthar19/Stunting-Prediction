# Script for extracting the monthly 
import pandas as pd

f_df = pd.read_excel("data/weight-Age/girls-zscore-weight-tables.xlsx")
m_df = pd.read_excel("data/weight-Age/boys-zscore-weight-tables.xlsx")

days_in_months = [x for x in range(0, 1856, 30)]
months = [i for i, v in enumerate(days_in_months)]
fL = []
fM = []
fS = []
mL = []
mM = []
mS = []

for day in days_in_months:
    fL.append(f_df[f_df["Day"]==day]["L"].item())
    fM.append(f_df[f_df["Day"]==day]["M"].item())
    fS.append(f_df[f_df["Day"]==day]["S"].item())
    
    mL.append(m_df[m_df["Day"]==day]["L"].item())
    mM.append(m_df[m_df["Day"]==day]["M"].item())
    mS.append(m_df[m_df["Day"]==day]["S"].item())

new_f_df = pd.DataFrame({
    "Day" : days_in_months,
    "Month" : months,
    "L" : fL,
    "M" : fM,
    "S" : fS
})

new_m_df = pd.DataFrame({
    "Day" : days_in_months,
    "Month" : months,
    "L" : mL,
    "M" : mM,
    "S" : mS
})

new_f_df.to_csv("data/weight-Age/Monthly-girls-weight-z-score.csv")
new_m_df.to_csv("data/weight-Age/Monthly-boys-weight-z-score.csv")
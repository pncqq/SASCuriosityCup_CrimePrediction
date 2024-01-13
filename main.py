import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Wczytywanie danych
file_path = 'communities.data'
column_names = [
    'state', 'county', 'community', 'communityname', 'fold', 'population', 'householdsize', 'racepctblack',
    'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
    'numbUrban', 'pctUrban',
    'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire',
    'medFamInc',
    'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap',
    'NumUnderPov',
    'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed', 'PctEmploy',
    'PctEmplManu',
    'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr',
    'FemalePctDiv',
    'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par',
    'PctWorkMomYoungKids',
    'PctWorkMom', 'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8',
    'PctImmigRec10',
    'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly',
    'PctNotSpeakEnglWell',
    'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
    'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup',
    'PctHousOwnOcc',
    'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb',
    'OwnOccLowQuart',
    'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent', 'MedRentPctHousInc',
    'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn',
    'PctBornSameState',
    'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop',
    'LemasSwFTFieldOps',
    'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop',
    'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian',
    'PctPolicMinor',
    'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans',
    'PolicCars',
    'PolicOperBudg', 'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop',
    'ViolentCrimesPerPop'
]

data = pd.read_csv(file_path, names=column_names)

# Konwersja '?' na NaN dla prawidłowego obsłużenia brakujących danych
data.replace('?', pd.NA, inplace=True)

# Brakujące dane
none_frequency = data.isna().mean() * 100

plt.figure(figsize=(15, 10))
none_frequency.plot(kind='bar')
plt.title('Percentage of None/NaN Values in Each Column')
plt.xlabel('Column')
plt.ylabel('Percentage of None/NaN Values')
plt.xticks(range(len(none_frequency)), [str(i + 1) for i in range(len(none_frequency))], rotation=90)
plt.show()


# Usunięcie kolumny 'communityname'
data = data.drop('communityname', axis=1)

# Zamiana na numeryczne
data = data.apply(pd.to_numeric, errors='coerce')

# Usuwanie kolumn z dużą ilością brakujących danych
threshold_for_missing_data = 0.5
missing_values = data.isna().sum()
columns_to_drop = missing_values[missing_values > len(data) * threshold_for_missing_data].index
data_cleaned = data.drop(columns=columns_to_drop)


# Usuwanie kolumny 'communityname'
y = data_cleaned['ViolentCrimesPerPop']
X = data_cleaned.drop('ViolentCrimesPerPop', axis=1)

# Podział danych na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

wiersz_130= ['6','?','?','Watsonvillecity',1,0.03,0.78,0.01,0.31,0.34,1,0.46,0.57,0.36,0.39,0.05,1,0.27,0.53,0.34,0.36,0.54,0.59,0.5,0.25,0.18,0.26,0.22,0.16,0.31,0.19,0.23,0.04,0.4,1,0.83,0.15,0.75,0.46,0.36,0.27,0.61,0.19,0.36,0.48,0.5,0.46,0.9,0.55,0.53,0.68,0.69,0.57,0.5,0.02,0.37,0.07,0.41,0.5,0.52,0.54,1,1,1,0.98,0,1,1,1,0.82,0.57,1,0.24,1,0.82,0,0.02,0.83,0.35,0.26,0.51,0.67,0.2,0.59,0.47,0.48,0.48,0.46,0.53,0.62,0.54,0.72,0.83,0.25,0.07,0.06,1,0.46,0.41,0.72,0.73,'?','?','?','?','?','?','?','?','?','?','?','?','?','?','?','?','?',0.01,0.44,0.11,'?','?','?','?',0,'?',0.62]
print(wiersz_130[31])
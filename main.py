import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns



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

# Wypisz kolumny do wyrzucenia
dropped_columns = columns_to_drop.tolist()
print(f'Dropped columns: communityname, {dropped_columns}')
print(f'Numbers of dropped columns: {len(dropped_columns) + 1}')

# # Usuwanie kolumny 'communityname'
y = data_cleaned['ViolentCrimesPerPop']
X = data_cleaned.drop('ViolentCrimesPerPop', axis=1)

# Podział danych na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Obliczanie korelacji ze zmienna docelową
feature_correlation = data_cleaned.corr()['ViolentCrimesPerPop'].sort_values()

# Zakładamy, że "niska" korelacja oznacza wartość bezwzględną poniżej pewnego progu, np. 0.1
low_correlation_threshold = 0.075
low_corr_features = feature_correlation[abs(feature_correlation) < low_correlation_threshold]
print("Cechy o niskiej korelacji z 'ViolentCrimesPerPop':\n", low_corr_features)

low_corr_feature_names = low_corr_features.index.tolist()
print("NAZWY Cechy o niskiej korelacji z 'ViolentCrimesPerPop':\n", low_corr_feature_names)


# Trenowanie modelu losowego lasu
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Ważności cech
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Zakładamy, że "niska" ważność oznacza wartość poniżej pewnego progu, np. 0.01
low_importance_threshold = 0.01
low_importance_features = feature_importances[feature_importances < low_importance_threshold]
print("Cechy o niskiej ważności:\n", low_importance_features)


unique_values_set = set(low_corr_features.index.tolist()) | set(low_importance_features.index.tolist())
unique_values_series = pd.Series(list(unique_values_set))

X_train_dropped = X_train.drop(unique_values_series, axis=1)
X_test_dropped = X_test.drop(unique_values_series, axis=1)



model = LinearRegression()
model.fit(X_train_dropped, y_train)

y_pred = model.predict(X_test_dropped)

mse = mean_squared_error(X_test_dropped, y_pred)
r2 = r2_score(y_test, y_pred)

print("Błąd średniokwadratowy (MSE):", mse)
print("Współczynnik determinacji (R^2):", r2)







# model = LinearRegression()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print("Błąd średniokwadratowy (MSE):", mse)
# print("Współczynnik determinacji (R^2):", r2)

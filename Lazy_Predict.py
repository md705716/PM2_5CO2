import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
PMCO2 = pd.read_csv('city_2.csv')
y1 = np.array(PMCO2.地表PM25质量浓度)
y2 = np.array(PMCO2.CO2排放量_吨)
y = np.vstack((y1, y2))
y=  y.T
from sklearn.preprocessing import normalize
y = normalize(y, axis=0)
PMCO2 = PMCO2.drop(['地表PM25质量浓度','CO2排放量_吨'], axis=1)
PMCO2 = PMCO2.fillna(PMCO2.mean())
X = normalize(PMCO2,axis = 0)
X = pd.DataFrame(X)
X = X.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=6)
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

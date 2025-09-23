import numpy as np
from sklearn import linear_model

# Данные
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 1.3, 3.75, 2.25])

# Линейная регрессия
reg = linear_model.LinearRegression()
reg.fit(X, y)

print("Коэффициенты:", reg.coef_)
print("Свободный член:", reg.intercept_)
print("Предсказание для x=6:", reg.predict([[6]]))
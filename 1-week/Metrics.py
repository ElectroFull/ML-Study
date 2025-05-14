# Метрики регрессии
 

# Ограничение времени	1 секунда
# Ограничение памяти	64Mb
# Ввод	стандартный ввод или input.txt
# Вывод	стандартный вывод или output.txt
 
# Используя только библиотеки numpy и math, реализуйте метрики регрессии на python. Способ реализации (одна функция на каждую метрику или все в одной) остается на ваше усмотрение.

# Функция принимает на вход две выборки — истинные значения целевого признака и предсказанные.

# Формат ввода
# на первой строке через пробел истинные значения, на второй — предсказанные.

# Формат вывода
# Метрики в формате 2-х значений после запятой.

# Пример
# Ввод	Вывод
# 1 5 2 9 2 3
# 1.5 5.2 2.3 8.7 2.8 3.3
# MSE: 0.20
# MAE: 0.40
# RMSE: 0.45
# Примечания
# Использование библиотеки sklearn запрещено.



import numpy as np


def MSE(y: np.ndarray, y_: np.ndarray) -> float:  # Mean Squared Error
    return (np.sum((y - y_) ** 2)) / len(y)


def MAE(y: np.ndarray, y_: np.ndarray) -> float:  # Mean Absolute Error
    return np.sum(np.abs(y - y_)) / len(y)


def RMSE(y: np.ndarray, y_: np.ndarray) -> float:  # Square Root Mean Squared Error
    return np.sqrt(np.sum((y - y_) ** 2) / len(y))


y = np.fromstring(input(), sep=" ", dtype="float64")
y_ = np.fromstring(input(), sep=" ", dtype="float64")

print(f"MSE: {MSE(y, y_):.2f}")
print(f"MAE: {MAE(y, y_):.2f}")
print(f"RMSE: {RMSE(y, y_):.2f}")

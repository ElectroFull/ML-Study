# MinMax Scaler
# Ограничение времени	1 секунда
# Ограничение памяти	64Mb
# Ввод	стандартный ввод
# Вывод	стандартный вывод
# Реализуйте преобразование MinMaxScaler "своими руками", повторяющее работу метода из https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

# Использовать готовый метод запрещается.

# Напишите функцию minmax_scale, которая принимает на вход матрицу признаков и выдает отмасштабированную матрицу.

# *обратите внимание на случай, когда фича принимает единственное значение

# Пример
# Ввод	Вывод
# X = np.array([[1, 2], [2, 1]])
# print(minmax_scale(X))
# [[0. 1.]
#  [1. 0.]]
# Примечания
# В файле, сдаваемом в тестирующую систему, не должно быть ничего, кроме класса и, возможно, вспомогательных функций.


import numpy as np


def minmax_scale(X: np.ndarray) -> np.ndarray:
    x_min = np.min(X, axis=0, keepdims=True)
    x_max = np.max(X, axis=0, keepdims=True)
    d = x_max - x_min
    d[d == 0] = 1
    return (X - x_min) / d
# Змейка m x n
# Ограничение времени	1 секунда
# Ограничение памяти	64.0 Мб
# Ввод	стандартный ввод или input.txt
# Вывод	стандартный вывод или output.txt
# Напишите функцию  
#  snake(m,n), которая принимает в себя размеры матрицы, а возвращает матрицу размера  
#  m×n, заполненную «змейкой» числами от  
# 1 до m∗n: нечётные стороны слева направо, а чётные — справа налево. Матрица должна быть типа  
#  numpy.ndarray.

# Пример
# Ввод	Вывод
# print(snake(3, 4))
# [[ 1  2  3  4]
#  [ 8  7  6  5]
#  [ 9 10 11 12]]

import numpy as np


def func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * len(y[0]) * (x % 2 == 0) + (y + 1) * (x % 2 == 0) + ((x + 1) * len(y[0]) - y) * (x % 2)


def snake(m: int, n: int) -> np.ndarray:
    arr = np.zeros((m, n))
    return np.fromfunction(func, (m, n), dtype="int")
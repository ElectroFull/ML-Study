# K ближайших пингвинов
# Ограничение времени	5 секунд
# Ограничение памяти	128Mb
# Ввод	стандартный ввод или input.txt
# Вывод	стандартный вывод или output.txt
# Реализуйте нахождение  
# k
#   ближайших соседей по евклидову расстоянию для датасета penguins. Библиотека sklearn отключена. На входе два числа: номер строки n ( 
# 0
# ≤
# n
# ≤
# 3
# 3
# 2
#  ) и число соседей  
# k
# ≥
# 1
#  . . На выходе — k ближайших соседей для этой строки, каждый в формате numpy.ndarray и с новой строки. Подробный план:
# Импортируйте необходимые далее библиотеки.
# Загрузите датасет ’penguins.csv’. В проверяющей системе он находится в текущем каталоге (pd.read_csv(‘penguins.csv’) сработает), для отладки можно использовать датасет по ссылке.
# Удалите пустые строки.
# Скопируйте столбцы bill_length_mm, bill_depth_mm в новую таблицу (порядок столбцов важен). Переведите эту таблицу в numpy.ndarray.
# Напишите функцию нахождения евклидова расстояния.
# Напишите функцию, которая находит  
# k
#   ближайших соседей к строке под номером  
# n
#  . Соседи должны идти в порядке возрастания расстояния.
# Выведите полученные строки, каждая с новой строки. Самой строки под номером  
# n
#   не должно быть в этом списке.
# Формат ввода
# Два целых числа на двух строчках
# Пример 1
# Ввод	Вывод
# 0
# 5
# [39.  18.7]
# [39.2 18.6]
# [38.9 18.8]
# [38.7 19. ]
# [39.6 18.8]
# Пример 2
# Ввод	Вывод
# 302
# 3
# [47.7 15. ]
# [47.8 15. ]
# [47.3 15.3]


import numpy as np
import pandas as pd
import math


def euclidian_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return math.sqrt(np.sum((vector_a - vector_b) ** 2))


def knn(vector_s: np.ndarray, k: int, n: int) -> np.ndarray:
    distances = np.array([euclidian_distance(x, vector_s[n]) for x in vector_s])
    return vector_s[np.argsort(distances)[1:k + 1]]


data = pd.read_csv("penguins.csv")
data = data.dropna()

features = data[["bill_length_mm", "bill_depth_mm"]].to_numpy()

n, k = map(int, [input(), input()])

res = knn(features, k, n)

print("\n".join(str(v) for v in res))

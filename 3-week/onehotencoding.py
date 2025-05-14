# One hot encoding
# Ограничение времени	1 секунда
# Ограничение памяти	64Mb
# Ввод	стандартный ввод
# Вывод	стандартный вывод
# Реализуйте упрощенный вариант преобразования one_hot_encoding, не используя pd.get_dummies и sklearn.preprocessing.OneHotEncoder.

# Напишите функцию onehot_encoding, которая принимает на вход одномерный массив numpy размера (число объектов) со значениями категориального признака и выдает целочисленную матрицу размера (число объектов * число значений фичи), заполненную как one_hot_encoding.

# Бинарные вектора в матрице идут в порядке сортировки значений категориального признака. Те если признак принимает значения b, a, с, то самым левым в one_hot матрице будет вектор для a, а самым правым для c.

# Пример
# Ввод	Вывод
# x = np.array([3, 2, 2, 1])
# print(onehot_encoding(x))
# [[0 0 1]
#  [0 1 0]
#  [0 1 0]
#  [1 0 0]]
# Примечания
# В файле, сдаваемом в тестирующую систему, не должно быть ничего, кроме класса и, возможно, вспомогательных функций.


import numpy as np


def onehot_encoding(x: np.ndarray) -> np.ndarray:
    unique_cnt = np.unique(x)
    encoding = np.zeros((len(x), len(unique_cnt)), dtype="int")
    compression = dict()
    for id, i in enumerate(unique_cnt):
        compression[i] = id
    for id, category in enumerate(x):
        encoding[id, compression[category]] = 1
    return encoding
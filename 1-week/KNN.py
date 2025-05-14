# KNN на практике
 

# Ограничение времени	5 секунд
# Ограничение памяти	128Mb
# Ввод	стандартный ввод или input.txt
# Вывод	стандартный вывод или output.txt

# Импортируйте необходимые далее библиотеки.
# Загрузите датасет ’penguins.csv’. В проверяющей системе он находится в текущем каталоге (pd.read_csv(‘penguins.csv’) сработает), для отладки можно использовать датасет по ссылке.
# Удалите пустые строки.
# Скопируйте столбец species в переменную labels, а столбцы bill_length_mm, bill_depth_mm — в переменную features.
# Скопируйте эту строку для разбиения на тестовую и обучающую выборку. Важно зафиксировать random_state, чтобы ответ был одинаковым на всех машинах:
#         train_features, test_features, train_labels, test_labels = train_test_split( 
#             features, labels, test_size=0.2, random_state=123)
# На этих выборках обучите 20 моделей следующим образом. Переберите количество соседей от 1 до 10, для каждого значения используйте два значения параметра weights: "uniform "distance". Зафиксируйте параметры, при которых получилось лучшее и худшее значение accuracy.
# Выведите на экран значения accuracy для лучшей и худшей модели в следующем формате:
#         print(’Best accuracy: {:.6f}’.format(...)) 
#         print(’Worst accuracy: {:.6f}’.format(...))
# Вместо многоточий ваши значения.

# Код без обучения моделей не будет засчитан. Оставьте два print() с выводом лучшей и худшей метрики, остальные надо убрать или закомментировать.



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('penguins.csv')
data = data.dropna()

labels = data["species"].to_numpy()
features = data[["bill_length_mm", "bill_depth_mm"]].to_numpy()

train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                            test_size=0.2,
                                                                            random_state=123)

best_accuracy, worst_accuracy = 0.0, 1.0

for i in range(1, 11):
    for weights in ("uniform", "distance"):
        knn_object = KNeighborsClassifier(n_neighbors=i, weights=weights)
        knn_object.fit(train_features, train_labels)
        predictions = knn_object.predict(test_features)

        accuracy = accuracy_score(test_labels, predictions)
        best_accuracy = max(best_accuracy, accuracy)
        worst_accuracy = min(worst_accuracy, accuracy)

print("Best accuracy: {:.6f}".format(best_accuracy))
print("Worst accuracy: {:.6f}".format(worst_accuracy))

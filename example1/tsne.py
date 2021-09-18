# Импорт библиотек
import numpy
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

fname = 'C:/Users/sinhro/Documents/vsu/sem7/ИнтСиТ/l1/python/taskdata/housing.data'
import pandas as pd

# reading csv files
dataRaw = pd.read_csv(fname, sep="\t")
data = []
for i in range(0, dataRaw.__len__()):
    col = dataRaw.values[i][0].split()
    data.append(col)
print(data)

arr = numpy.array(data)


# Загрузка датасета
iris_df = datasets.load_iris()

# Определяем модель и скорость обучения
model = TSNE(learning_rate=100)

# Обучаем модель
# transformed = model.fit_transform(iris_df.data)
transformed = model.fit_transform(arr)

# Представляем результат в двумерных координатах
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]

# plt.scatter(x_axis, y_axis, c=iris_df.target)
plt.scatter(x_axis, y_axis)
plt.show()

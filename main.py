# import statements
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt  # create blobs

FILE_NAME = 'C:/Users/sinhro/Documents/vsu/sem7/ИнтСиТ/l1/python/taskdata/housing.data'


def loadData(filename='C:/Users/sinhro/Documents/vsu/sem7/ИнтСиТ/l1/python/taskdata/housing.data'):
    import pandas as pd
    import numpy

    # reading csv files
    dataRaw = pd.read_csv(filename, sep="\t")
    data = []
    for i in range(0, dataRaw.__len__()):
        col = dataRaw.values[i][0].split()
        data.append(col)
    # print(data)
    arr = numpy.array(data)

    return arr


def normData(data: list):
    from sklearn import preprocessing
    dataNorm = preprocessing.MinMaxScaler().fit_transform(data)
    return dataNorm


def test():
    data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6,
                      random_state=50)  # create np array for data points
    points = data[0]  # create scatter plot
    plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='viridis')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)


def alg(input_data, nClusters=3, x_axis_num=0, y_axis_num=1):
    # sortedByXData = sorted(input_data, key=lambda row: row[x_acis_num])
    # data = sortedByXData
    data = input_data
    print("     Length of data:", len(data))
    print("     First 5 rows of data:")
    print(data[0:5])
    print("     Last 5 rows of data:")
    print(data[-5:])

    # Импортируем библиотеки
    from sklearn import datasets
    from sklearn.cluster import KMeans

    # Описываем модель
    model = KMeans(n_clusters=nClusters)

    # Проводим моделирование
    # model.fit(data)

    # Предсказание на всем наборе данных
    labels = model.fit_predict(data)

    # Выводим предсказания
    print(labels)

    x_axis = data[:, x_axis_num]
    y_axis = data[:, y_axis_num]
    plt.scatter(x_axis, y_axis, c=labels)
    plt.show()


loadedFromFile = loadData()

# data = loadedFromFile[:, [5, 9]]

# Убираем лишнее поле
data = loadedFromFile[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

# Выполняем стандартизацию
norm_data = normData(data)

# Основной алгоритм
alg(norm_data, 2, 5, 9)

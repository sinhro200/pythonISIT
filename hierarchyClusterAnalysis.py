from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
seeds_df = pd.read_csv('seeds-less-rows.csv') # Сеть можно скачать
seeds_df.head()

varieties = list(seeds_df.pop('grain_variety'))
varieties
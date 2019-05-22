# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:28:14 2019

@author: Ishan Yash
"""

dd = read_ucr_data(r'G:\Coding\ML\UCRArchive_2018\Car\Car_TRAIN.tsv')

win = 30
paa = 6
alp = 6
na_strategy = "exact"
ztresh = 0.01

bags = {}

for key, arr in dd.items():
    print(key)
    bags[key] = manyseries_to_wordbag(dd[key], win, paa, alp, na_strategy, ztresh)

[*bags.copy()]
vectors = bags_to_tfidf(bags)
vectors['classes']

dt = read_ucr_data(r'G:\Coding\ML\UCRArchive_2018\Car\Car_TEST.tsv')

import matplotlib.pyplot as plt
import datetime
import numpy as np

series = dt['1'][6]
x = np.arange(0, len(series))
y = np.asarray(series)

plt.plot(x,y)
plt.show()

test_bag = series_to_wordbag(series, 30, 6, 6, "exact", 0.01)
res = cosine_similarity(vectors, test_bag)
class_for_bag(res)

for cls in [*dt.copy()]:
    print(cls)
    i = 0
    for s in dt[cls]:
        sim = cosine_similarity(vectors, 
                                series_to_wordbag(s, 30, 6, 6, "none", 0.01))
        res = class_for_bag(sim)
        if res != cls:
            print(" misclassified", i, "as", res, sim)
        i = i + 1


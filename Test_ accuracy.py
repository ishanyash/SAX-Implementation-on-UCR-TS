# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:41:48 2019

@author: Ishan Yash
"""
train = read_ucr_data(r'G:\Coding\ML\UCRArchive_2018\CBF\CBF_TRAIN.tsv')
test = read_ucr_data(r'G:\Coding\ML\UCRArchive_2018\CBF\CBF_TEST.tsv')

win = 30
paa = 6
alp = 6
na_strategy = "exact"
zthresh = 0.01

def test_accuracy(dd_train, dd_test, sax_win, sax_paa, sax_alp, sax_strategy, z_threshold):
    
    train_bags = {}
    for key, arr in dd_train.items():
        train_bags[key] = manyseries_to_wordbag(dd[key], sax_win, sax_paa,
                                                sax_alp, sax_strategy, z_threshold)
    
    tfidf_vectors = bags_to_tfidf(train_bags)

    correct = 0
    count = 0

    for cls in [*dd_test.copy()]:
        for s in dd_test[cls]:
            sim = cosine_similarity(tfidf_vectors, 
                                    series_to_wordbag(s, sax_win, sax_paa,
                                                      sax_alp, sax_strategy, z_threshold))
            res = class_for_bag(sim)
            if res == cls:
                correct = correct + 1
            count = count + 1
    
    return correct / count

accuracy = test_accuracy(train, test, win, paa, alp, na_strategy, zthresh)
errorr = 1 - accuracy

print('Accuracy:',accuracy)
print('Error Rate: ', errorr)
#print('Error Rate: ', 1 - accuracy)



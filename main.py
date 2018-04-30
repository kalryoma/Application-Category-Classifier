import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from math import exp

def get_label(label_file, chunksize):
    label_list = set()
    label_dict = dict()
    label_name = set()
    for chunk in pd.read_csv(label_file, header=None, chunksize=3000):
        for row in chunk.values:
            name = row[0]
            label = row[1]
            label_name.add((name, label))
            if label not in label_list:
                label_list.add(label)
    label_list = list(label_list)
    label_list.sort(key=str.lower)
    for name, label in label_name:
        label_dict[name] = label_list.index(label)
    return label_list, label_dict

def get_data(label_file, data_file, chunksize):
    label_list, label_dict = get_label(label_file, chunksize)
    label_list_length = len(label_list)
    x = []
    y = []
    for chunk in pd.read_csv(data_file, header=None, chunksize=chunksize):
        for row in chunk.values:
            name = row[0]
            t = label_dict[name]
            t = np.around(t, 8)
            y.append(t)
            data = np.array(row[1:]).astype(np.float)
            data = np.around(data, 8)
            x.append(data)
    x = csr_matrix(np.asmatrix(x))
    return x, y, label_list_length

def generate_regularisaztion_mat(s, lmd):
    s_inv = 1.0/(s+lmd)
    return csr_matrix(np.diag(s_inv))

def get_w(x, y, lmd, k):
    w = x.T.dot(x)
    u, s, ut = svds(w, k)
    u = csr_matrix(np.around(u, 10))
    ut = csr_matrix(np.around(ut, 10))
    v = generate_regularisaztion_mat(s, lmd)
    w = u.dot(v).dot(ut)
    t = csr_matrix(x.T.dot(y))
    w = np.dot(w, t)
    return w

def get_category_index(label_list_length, predicted_result):
    predicted_label = []
    for i in range(len(predicted_result)):
        yi = predicted_result[i][0]
        this_label = int(round(yi))
        predicted_label.append(this_label)
    return predicted_label

def calc_accuracy(actual_t, label_list_length, predicted_result):
    predicted_label = get_category_index(label_list_length, predicted_result)
    trueNo = 0
    for i in range(len(actual_t)):
        if actual_t[i] == predicted_label[i]:
            trueNo += 1
        # print(actual_t[i], predicted_label[i], actual_t[i] == predicted_label[i])
    return trueNo*100.0/len(actual_t)

data_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/sample_data.csv'
label_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/training_labels.csv'
chunksize = 50
x, y, label_list_length = get_data(label_file, data_file, chunksize)

k_eig = range(100, 500, 50)
lmd = range(-10, 0)
for j in range (len(k_eig)):
    for i in range(len(lmd)):
        w = get_w(x, np.asmatrix(y).T, exp(lmd[i]), k_eig[j])
        predicted_result = x.dot(w)
        predicted_result = predicted_result.todense().tolist()
        accuracy = calc_accuracy(y, label_list_length, predicted_result)
        print(k_eig[j], lmd[i], accuracy)

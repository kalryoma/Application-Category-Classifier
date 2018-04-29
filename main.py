import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from math import exp, sqrt

def normalise(a, max):
    return a*1.0/max
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
    for name, label in label_name:
        label_dict[name] = label_list.index(label)
    return label_list, label_dict

def get_data(label_file, data_file, chunksize):
    label_list, label_dict = get_label(label_file, chunksize)
    label_list_length = len(label_list)
    x = []
    y = []
    x_max = 0
    for chunk in pd.read_csv(data_file, header=None, chunksize=chunksize):
        for row in chunk.values:
            name = row[0]
            t = np.zeros(13626)
            t[0] = normalise(label_dict[name], label_list_length-1)
            t = np.ndarray(shape=(13626, 1), buffer=t)
            t = csr_matrix(t)
            y.append(t)
            data = np.array(row[1:]).astype(np.float)
            data_max = np.max(data)
            if data_max > x_max:
                x_max = data_max
            data = np.around(data, 8)
            data = np.ndarray(shape=(13626, 1), buffer=data)
            data = csr_matrix(data)
            x.append(data)
    y = np.asarray(y)[np.newaxis]
    return np.asarray(x), y.T, label_list_length, x_max

def generate_feature_mat(x_max, x, p):
    result = []
    for xi in x:
        resi = []
        xi = normalise(xi, x_max)
        for i in range(0, p+1):
            yi = xi.power(i)
            resi.append(yi)
        result.append(resi)
    return np.asarray(result)

def multiply_row_column(a, b, bRecalc):
    result = 0
    for i in range(a.size):
        result += a[i].T.dot(b[i]) if bRecalc else a[i]*b[i]
    return result[0, 0] if bRecalc else result

def matrix_mul(a, b, bRecalc):  # bRecalc means whether to calculate w or calculate predicted_y
    l = a.shape[0]
    k = b.shape[1]
    result = []
    for i in range(l):
        resi = []
        for j in range(k):
            row = a[i,:]
            column = b[:,j]
            resij = multiply_row_column(row, column, bRecalc)
            resi.append(resij)
        result.append(resi)
    return np.asarray(result)

def generate_regularisaztion_mat(p, lmd):
    regularisation = []
    for i in range(p+1):
        regularisation.append(lmd)
    return np.diag(regularisation)

def get_w(ph, y, p, lmd):
    w = matrix_mul(ph.T, ph, True)+generate_regularisaztion_mat(p, lmd)
    w = np.linalg.pinv(w)
    t = matrix_mul(ph.T, y, True)
    print(t)
    return np.dot(w, t)

def get_category_index(label_list_length, predicted_result):
    predicted_label = []
    l = predicted_result.shape[0]
    for i in range(l):
        xi = predicted_result[i][0]
        min_dist = 100000
        this_label = 1
        for j in range(label_list_length):
            norm_j = normalise(j, label_list_length-1)
            dist = calc_edist(xi, norm_j)
            if dist < min_dist:
                min_dist = dist
                this_label = norm_j
        predicted_label.append(this_label)
    return predicted_label

def calc_edist(a, k):
    x1 = a[0, 0]
    distsum = (x1-k)**2
    data_length = len(a.data)
    if data_length==0:
        return sqrt(distsum)
    else:
        start = 1 if a.data[0] == x1 else 0
        for i in range(start, data_length):
            distsum += a.data[i]**2
        return sqrt(distsum)

def calc_accuracy(actual_t, label_list_length, predicted_result):
    predicted_label = get_category_index(label_list_length, predicted_result)
    trueNo = 0
    for i in range(len(actual_t)):
        try:
            actual = actual_t[i][0].data[0]
        except IndexError:
            actual = 0
        if actual == predicted_label[i]:
            trueNo += 1
        # print(actual, predicted_label[i], actual == predicted_label[i])
    return trueNo*100.0/len(actual_t)

data_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/sample_data.csv'
label_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/training_labels.csv'
chunksize = 50
x, y, label_list_length, x_max = get_data(label_file, data_file, chunksize)

p = 3
lmd = exp(-2)
ph = generate_feature_mat(x_max, x, p)
w = get_w(ph, y, p, lmd)
print(w)

predicted_result = matrix_mul(ph, w, False)
accuracy = calc_accuracy(y, label_list_length, predicted_result)
print(accuracy)

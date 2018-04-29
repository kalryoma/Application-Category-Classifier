import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def get_label(label_file, chunksize):
    label_list = set()
    label_dict = dict()
    label_name = set()
    for chunk in pd.read_csv(label_file, header=None, chunksize=chunksize):
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

def get_data(data_file, chunksize):
    x = []
    y = []
    for chunk in pd.read_csv(data_file, header=None, chunksize=chunksize):
        for row in chunk.values:
            name = row[0]
            t = np.zeros(13626)
            # t[0] = label_dict[name]
            t[0] = 1
            t = np.ndarray(shape=(13626, 1), buffer=t)
            t = csr_matrix(t)
            y.append(t)
            data = np.array(row[1:]).astype(np.float)
            data = np.around(data, 3)
            data = np.ndarray(shape=(13626, 1), buffer=data)
            data = csr_matrix(data)
            x.append(data)
    y = np.asarray(y)[np.newaxis]
    return np.asarray(x), y.T

def generate_feature_mat(x, p):
    result = []
    for xi in x:
        resi = []
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

def get_w(ph, y, regularisation):
    w = matrix_mul(ph.T, ph, True)
    w = w+regularisation
    w = np.linalg.pinv(w)
    t = matrix_mul(ph.T, y, True)
    w = np.dot(w, t)
    return w

def linear_regression(x, y, p, lmd):
    ph = generate_feature_mat(x, p)
    regularisation = generate_regularisaztion_mat(p, lmd)
    w = get_w(ph, y, regularisation)
    return w, ph

data_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/sample_data.csv'
label_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/sample_labels.csv'
chunksize = 50
label_list, label_dict = get_label(label_file, chunksize)
x, y = get_data(data_file, chunksize)

p = 3
lmd = 0.1
w, ph = linear_regression(x, y, p, lmd)
print(w)

result_test = matrix_mul(ph, w, False)

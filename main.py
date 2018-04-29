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

def get_data(label_file, data_file, chunksize):
    label_list, label_dict = get_label(label_file, chunksize)
    x = []
    y = []
    for chunk in pd.read_csv(data_file, header=None, chunksize=chunksize):
        for row in chunk.values:
            name = row[0]
            t = np.zeros(13626)
            # t[0] = label_dict[name]+1
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
    return np.asarray(x), y.T, len(label_list)

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

def get_class(label_list_length, predicted_result):
    predicted_label = []
    l = predicted_result.shape[0]
    for i in range(l):
        xi = predicted_result[i][0]
        min_dist = 100000
        this_label = 0
        for j in range(label_list_length):
            dist = calc_edist(xi, j+1)
            if dist < min_dist:
                min_dist = dist
                this_label = j+1
        predicted_label.append(this_label)
    return predicted_label

def calc_edist(a, j):
    x1 = a[0, 0]
    distsum = (x1-j)**2
    start = 1 if a.data[0] == x1 else 0
    for i in range(start, len(a.data)):
        distsum += a.data[i]**2
    return distsum

def calc_accuracy(actual_t, label_list_length, predicted_result):
    predicted_label = get_class(label_list_length, predicted_result)
    trueNo = 0
    for i in range(len(actual_t)):
        if y[i][0].data[0] == predicted_label[i]:
            trueNo += 1
        # print(y[i][0].data[0], predicted_label, y[i][0].data[0] == predicted_label[i])
    return trueNo*100.0/len(actual_t)

data_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/sample_data.csv'
label_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/sample_labels.csv'
chunksize = 50
x, y, label_list_length = get_data(label_file, data_file, chunksize)

p = 3
lmd = 0.1
w, ph = linear_regression(x, y, p, lmd)
print(w)

predicted_result = matrix_mul(ph, w, False)
accuracy = calc_accuracy(y, label_list_length, predicted_result)
print(accuracy)

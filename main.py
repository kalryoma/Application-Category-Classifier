import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from ridge_regression import Ridge

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
    y = np.asmatrix(y).T.astype(np.float)
    return x, y, label_list

def get_category_index(predicted_result):
    predicted_label = []
    for i in range(len(predicted_result)):
        yi = predicted_result[i]
        this_label = int(round(yi))
        predicted_label.append(this_label)
    return predicted_label

def calc_accuracy(actual_t, predicted_result):
    predicted_result = predicted_result.T.tolist()[0]
    predicted_label = get_category_index(predicted_result)
    trueNo = 0
    for i in range(len(actual_t)):
        if actual_t[i] == predicted_label[i]:
            trueNo += 1
        # print(actual_t[i], predicted_label[i], actual_t[i] == predicted_label[i])
    return trueNo*100.0/len(actual_t)

data_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/sample_data1.csv'
label_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/training_labels.csv'
chunksize = 50
X, y, label_list = get_data(label_file, data_file, chunksize)

ridge = Ridge(max_iters=4000, alpha=0.001, lmd=0.05)
ridge.regression(X, y)
outs = ridge.predict(X)
accuracy = calc_accuracy(y.T.tolist()[0], outs)
print(accuracy)

def cross_validation(k, x, y):
    datasize = int(len(y)/k)
    average_accuracy = 0.0
    start_row = 0
    for i in range(k):
        while start_row < len(y):
            end_row = start_row+datasize
            if end_row > len(y):
                end_row = len(y)
            x_training = csr_matrix(np.delete(x.todense(), range(start_row, end_row), 0))
            x_test = x[start_row:end_row]
            y_training = y[:start_row]+y[end_row:]
            y_test = y[start_row:end_row]
            start_row = end_row
        ridge.regression(x_training, np.asmatrix(y_training).T)
        outs = ridge.predict(x_test)
        accuracy = calc_accuracy(y_test, outs)
        print(accuracy)
        average_accuracy += accuracy
    return average_accuracy/(k*1.0)

vali = cross_validation(10, X, y.T.tolist()[0])
print(vali)

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix# from sklearn.naive_bayes import MultinomialNB
from MultinomialNB import MultinomialNB

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
    return x, y, label_list

def calc_accuracy(actual_t, predicted_label):
    trueNo = 0
    for i in range(len(actual_t)):
        if actual_t[i] == predicted_label[i]:
            trueNo += 1
    return trueNo*100.0/len(actual_t)

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

            model = MultinomialNB()
            model.fit(x_training, np.array(y_training))
            outs = model.predict(x_test)
            accuracy = calc_accuracy(y_test, outs)
            print(accuracy)
            average_accuracy += accuracy
    return average_accuracy/(k*1.0)

data_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/sample_data1.csv'
label_file = '/Users/kalryoma/Downloads/5318Assignment1_Data/training_labels.csv'
chunksize = 50
x, y, label_list = get_data(label_file, data_file, chunksize)

validation = cross_validation(5, x, y)
print(validation)

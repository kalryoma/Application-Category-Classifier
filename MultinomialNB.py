import numpy as np
from scipy.sparse import csr_matrix

class MultinomialNB():

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, x, y):
        label_list = set()
        for i in y:
            if i not in label_list:
                label_list.add(i)
        label_list = sorted(list(label_list))
        for i in range(len(y)):
            y[i] = label_list.index(y[i])

        Pc = np.zeros(len(label_list))
        for i in y:
            Pc[i] += 1
        Pc = np.log(Pc/(1.0*len(y)))
        self.Pc = Pc
        
        alpha = self.alpha
        Pac = np.zeros((len(Pc), x.shape[1]))
        (row, col) = x.nonzero()
        for i in range(len(row)):
            Pac[y[row[i]]][col[i]] += 1
        for i in range(len(Pc)):
            Pac[i] = (Pac[i]+alpha)*1.0/(sum(Pac[i])+x.shape[1]*alpha)
            Pac[i] = np.log(Pac[i])
        Pac = csr_matrix(Pac)
        self.Pac = Pac

    def predict(self, test):
        Pc = self.Pc
        Pac = self.Pac
        log_prob = test.dot(Pac.T)
        log_prob += Pc
        outs = np.argmax(log_prob, axis=1)
        return outs.T.tolist()[0]

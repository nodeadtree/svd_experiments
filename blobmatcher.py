import numpy as np
import argparse


class blob_model(object):
    def __init__(self):
        pass

    def fit(self, vectors, labels):
        self.dataset = dict()
        self.model = dict()
        for i, j in zip(vectors, labels):
            if j not in self.dataset:
                self.dataset[j] = []
            self.dataset[j].append(i)
        for i in self.dataset:
            array = np.array(self.dataset[i])
            self.model[i] = np.linalg.svd(array)[1:]
            print(np.shape(self.model[i][1]))


    def predict(self, vectors=None):
        if vectors is None:
            print('Running training set as test case')
            print('--------------')
            successes = 0
            for i in self.dataset:
                print('true label value is : ' + str(i))
                pre = self.predict(self.dataset[i])
                print('prediction is : ' + str(pre))
                print(i)
                print(pre)
                print('--------------')
                if i == pre:
                    successes +=1
            else:
                print("number of successes was : " + str(successes))
                print("Overall accuracy was : " +  str(successes / float(len(self.dataset))))
                return
        sample = np.linalg.svd(vectors)[1:]
        blob_info = list()
        for i in self.model:
            x=0
            for k in range(np.shape(sample[1])[1]):
                x += ((sample[0][k] * np.dot(sample[1][:,k], self.model[i][1][:,k]))-self.model[i][0][k])**2
            blob_info.append((x,i))
        return(min(blob_info, key=lambda x: x[0])[1])

    def set_dataset(self, vectors, labels):
        self.dataset=dict()
        for i, j in zip(vectors, labels):
            if j not in self.dataset:
                self.dataset[j] = []
            self.dataset[j].append(i)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-train', help="training input file")
    arg_parser.add_argument('-test', help="test input file")
    a = arg_parser.parse_args()
    
    train_data = np.genfromtxt(a.train, dtype=None)
    b = blob_model()
    b.fit(train_data[:,1:9], train_data[:,0])

    test_data = np.genfromtxt(a.test, dtype=None)
    b.set_dataset(test_data[:,1:9], test_data[:,0])
    b.predict()
    









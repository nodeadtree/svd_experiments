import numpy as np
import argparse
import random

class blobmo(object):

    def __init__(self):
        pass

    #Fit method
    #Accepts list of vectors, and list of labels associated with these vectors
    def fit(self, vectors, labels):
        dataset = dict()
        self.model = dict()
        for i,j in zip(vectors, labels):
            if j not in dataset:
                dataset[j] = list()
            dataset[j].append(i)
        for i in dataset:
            dataset[i] = self.normalize(dataset[i])
            self.model[i] = np.linalg.svd(np.array(dataset[i]).T)
            self.model[i] = [np.matrix(self.model[i][0]), self.model[i][1]]

    #Makes prediction on a matrix
    def predict(self, vectors):
        projections = list()
        vectors = self.normalize(vectors)
        sample = np.linalg.svd(vectors.T)
        singular_vectors = np.matrix(sample[0])
        for i in self.model:
            x = 0
            for j in range(np.shape(singular_vectors)[0]):
                x += (sample[1][j] * np.absolute(np.dot(singular_vectors[:,j].T,self.model[i][0][:,j])) - self.model[i][1][j])**2
            projections.append((i,x))
        prediction = min(projections, key=lambda x:x[1][0,0])[0]
        return(prediction)

    #Runs evaluation with a full list of vectors
    def score(self, vectors, labels):
        pass

    #Normalizing function
    def normalize(self, vectors):
        average_length = 0
        total = 0
        for i in vectors:
            average_length += np.linalg.norm(i)
            total += 1
        average_length = average_length / total
        nu_vectors = list()
        for i in vectors:
            nu_vectors.append((i/average_length)-1)
        nu_vectors = np.array(nu_vectors)
        return nu_vectors


        

        

class vector_bundler(object):

    def __init__(self, data, labels, bundle_size):
        self.dataset = dict()
        self.bundle_size = bundle_size
        for i,j in zip(data, labels):
            if j not in self.dataset:
                self.dataset[j] = list()
            self.dataset[j].append(i)
        for i in self.dataset:
            trim_amount = (len(self.dataset[i]) % bundle_size)
            if trim_amount > 0:
                self.dataset[i] = self.dataset[i][:-trim_amount]

    def get_bundle(self):
        keys = list(self.dataset.keys())
        if len(keys) == 0:
            return None
        label = random.choice(keys)
        bundle = self.dataset[label][:self.bundle_size]
        self.dataset[label] = self.dataset[label][self.bundle_size:]
        if len(self.dataset[label]) < self.bundle_size:
            self.dataset.pop(label)
        return (label, np.array(bundle))


if __name__ == "__main__":
    #argument stuff
    arg_parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-train', help="training input file")
    arg_parser.add_argument('-test', help="test input file")
    a = arg_parser.parse_args()

    #classifier stuff
    classifier = blobmo()
    
    #testing methods
    train_data = np.genfromtxt(a.train)
    train_data[:,0] = np.int_(train_data[:,0])

    test_data = np.genfromtxt(a.test)
    test_data[:,0] = np.int_(test_data[:,0])
    v = vector_bundler(test_data[:,1:9], test_data[:,0], 40)
    classifier.fit(train_data[:,1:9], train_data[:,0])
    correct = 0
    total = 0
    while True:
        x = v.get_bundle()
        if x is None:
            break
        if classifier.predict(x[1])==x[0]:
            correct +=1
        total +=1
    print("Accuracy : " + str(correct/total))




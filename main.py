import pandas as pd
import copy
import random
import xlrd
import numpy as np


class MLPPSO:
    def __init__(self, architect, individuals):
        self.architect = architect
        self.individuals = individuals
        self.initWeight = []

        for i in range(1, len(architect) - 1):
            perceptron_curr = architect[i]
            perceptron_prev = architect[i - 1]

        perceptron_currO = architect[i + 1]
        perceptron_prevO = architect[i]
        for ind in self.individuals:
            weight_prev = []
            weight_curr = []
            for c in range(perceptron_curr):
                weight_prev.append([])
                for p in range(perceptron_prev):
                    weight_prev[-1].append(ind[p + (c * perceptron_prev)])
            for p in range(perceptron_currO):
                weight_curr.append([])
                for c in range(perceptron_prevO):
                    weight_curr[-1].append(ind[(perceptron_prev * perceptron_curr) + (p + (p * perceptron_curr))])

        self.initWeight_Cur = weight_curr
        self.initWeight_Prev = weight_prev

    def train(self, X, Y):
        errors = []
        self.weight_curr = copy.deepcopy(self.initWeight_Cur)
        self.weight_prev = copy.deepcopy(self.initWeight_Prev)
        error = []
        for i in range(int(X.shape[0])):
            Alltest = [X[i]]
            yh = []
            y = []
            o = []
            out = []
            for j in range(len(self.weight_prev)):
                y.append(np.dot(Alltest, self.weight_prev[j]))
                activate_y = np.tanh(y)
                yh.append(activate_y)

            for k in range(len(self.weight_curr)):
                o = (np.asarray(yh) * np.asarray(self.weight_curr[k]))
                activate_o = np.tanh(sum(o[-1]))
                out.append(activate_o)

            error.append(out)
        error = np.array(error).reshape(len(error))
        errors.append(findMAE(error, Y))
        return errors


def findMAE(error, Y):
    return sum(abs(Y - error)) / len(Y)


def main():
    fitavg = 0.0
    avgerror = 0.0
    num_individual = 100
    # data_set = pd.read_excel("AQPredict5hr.xlsx").values
    data_set = pd.read_excel("AQPredict10hr.xlsx").values
    trainX = data_set[:, :-1].tolist()
    trainY = data_set[:, 8].tolist()

    p = random.uniform(0.1, 3)

    findMinX = np.min(trainX, 0)
    findMaxX = np.max(trainX, 0)
    findMinY = np.min(trainY, 0)
    findMaxY = np.max(trainY, 0)
    normX = np.divide(np.subtract(trainX, findMinX), np.subtract(findMaxX, findMinX)).tolist()
    normY = np.divide(np.subtract(trainY, findMinY), np.subtract(findMaxY, findMinY)).tolist()

    for c in range(0, 10):
        individuals = []
        pbest = []
        architect = [8, 10, 1]
        print("--------------Round {}-------------".format(c + 1))
        part = int(round(len(normX) / 10.0, 0))
        last = (c * part + part) - 1

        if c == 9:
            last = len(normX) - 1

        tmpX = []
        tmpY = []
        strainX = copy.deepcopy(normX)
        strainY = copy.deepcopy(normY)
        for i in range(last, c * part - 1, -1):
            tmpX.append(strainX[i])
            tmpY.append(strainY[i])
            strainX.pop(i)
            strainY.pop(i)
        rTestX = np.array(tmpX)
        rTestY = np.array(tmpY)
        rTrainX = np.array(strainX)
        rTrainY = np.array(strainY)

        for i in range(num_individual):
            individuals.append([])
            for j in range(architect[1] + 1):
                for k in range(architect[0]):
                    individuals[-1].append(random.uniform(-1, 1))

            pbest.append([num_individual, individuals[i]])
        individuals = np.asarray(individuals)
        v = np.asarray(individuals)
        mlp = MLPPSO(architect, individuals)
        for iter in range(10):
            fitness = mlp.train(rTrainX, rTrainY)
            for i in range(len(fitness)):
                if fitness[i] < pbest[i][0]:
                    pbest[i] = [fitness[i].copy(), individuals[i].copy()]
            for j in range(1, len(v)):
                v[j] = v[j - 1] + p * (pbest[j][1] - individuals[j])
                individuals[i] += v[j]

        fitness = mlp.train(rTestX, rTestY)
        print(fitness)
        fitavg += fitness[0]
    print("Average MAE: ", fitavg / 10)


if __name__ == '__main__':
    main()

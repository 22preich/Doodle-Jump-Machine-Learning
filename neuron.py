import numpy as np
import random

def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, inshape):
        self.weights = [1] * inshape
        self.bias = 0

    def init(self, newweights=0, newbias=0):
        self.weights = newweights if newweights else self.weights
        self.bias = newbias

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        # print(total)
        return sigmoid(total)


# weights = np.array([1, -2]) # w1 = 0, w2 = 1
# bias = 0                   # b = 0
n = Neuron(1)


def test_func(x):
    return 1 if x > 73 else 0


def test_acc(neuron, iterations):
    acc = 0
    for i in range(iterations):
        val = random.randrange(-100, 100)
        neuronres = neuron.feedforward(val)
        realres = test_func(val)
        if round(neuronres[0]) == realres:
            acc += 1
    return acc / iterations


print(test_acc(n, 1000))


def train(neuron, epochs):
    search = 10
    bias = 0
    direction = 0
    for _ in range(epochs):
        new_direction = 0
        neuron.init(newbias=bias)
        initial_acc = test_acc(neuron, 10000)
        if initial_acc == 1:
            print(f"FOUND ACCURATE BIAS: {bias}")
            break

        plus_bias = bias
        plus_bias += search

        minus_bias = bias
        minus_bias -= search

        neuron.init(newbias=plus_bias)
        plus_acc = test_acc(neuron, 10000)

        neuron.init(newbias=minus_bias)
        minus_acc = test_acc(neuron, 10000)


        """if plus_acc == initial_acc == minus_acc:
            search *= 2
            print("bias change had no effect on accuracy, expanding search")
            continue"""

        if plus_acc > initial_acc:
            new_direction = 1
        if minus_acc > initial_acc:
            new_direction = -1

        if new_direction and not direction:
            direction = new_direction
        elif direction is not new_direction:
            search = search * 0.9
            direction = new_direction if new_direction else direction

        bias += search * direction
        print(search, bias, direction)
    return test_acc(n, 1000)


print(train(n, 10000))

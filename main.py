# Importing packages
from matplotlib import pyplot as plt
import numpy as np
import random
import utils
from matplotlib import pyplot

def score(weights, bias, features):
    return features.dot(weights) + bias

def step(x):
    if x >= 0:
        return 1
    else:
        return 0

def prediction(weights, bias, features):
    scored = score(weights, bias, features)
    return step(scored)

# оцениваем ошибку
def error(weights, bias, features, label):
    pred = prediction(weights, bias, features)
    if pred == label:
        return 0
    else:
        scored = score(weights, bias, features)
        return np.abs(scored)

# оцениваем ошибку перцептрона
def mean_perceptron_error(weights, bias, features, labels):
    total_error = 0
    for i in range(len(features)):
        total_error += error(weights, bias, features[i], labels[i])
    return total_error/len(features)

# Shorter version of the perceptron trick
def perceptron_trick(weights, bias, point_coordinates, label, learning_rate = 0.01):
    #делаем предсказние значения в точке, если скаляное произведени двух весов и двух координат плюс bias больше 0, то 1 иначе 0
    pred = prediction(weights, bias, point_coordinates)
    #вычисляем новые коеффициенты и точку пересечения с осью.
    for i in range(len(weights)):
        weights[i] += (label-pred) * point_coordinates[i] * learning_rate
        bias += (label-pred)*learning_rate
    return weights, bias

def perceptron_algorithm(features, labels, learning_rate = 0.01, epochs = 200):
    weights = [1.0 for i in range(len(features[0]))]
    bias = 0.0
    errors = []
    for epoch in range(epochs):
        # Coment the following line to draw only the final classifier
        utils.draw_line(weights[0], weights[1], bias, color='grey', linewidth=1.0, linestyle='dotted')
        error = mean_perceptron_error(weights, bias, features, labels)
        errors.append(error)
        #получаем случайный индекс из диапазона длины массива точек.
        i = random.randint(0, len(features)-1)
        #в точке выбранной случайно делаем предсказание и корректируем коэффициенты. коэффициентов 2, координат в точке 2
        weights, bias = perceptron_trick(weights, bias, features[i], labels[i])
    utils.draw_line(weights[0], weights[1], bias)
    utils.plot_points(features, labels)
    plt.show()
    plt.scatter(range(epochs), errors)
    return weights, bias

features = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[2,3],[3,2]])
# meanings
labels = np.array([0,0,0,0,1,1,1,1])

# Plotting the points
utils.plot_points(features, labels)

# Uncomment the following line to see a good line fit for this data.
#utils.draw_line(1,1,-3.5)

#weights = [1,1]
#bias = -3.5
bias = -4

random.seed(0)
perceptron_algorithm(features, labels)

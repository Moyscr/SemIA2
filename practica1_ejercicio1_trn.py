import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()

    def unit_step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.unit_step_function(summation)

    def train(self, training_inputs, labels, max_epochs):
        for epoch in range(max_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

def read_training_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        labels = []
        for line in lines:
            values = line.strip().split(',')
            data.append([float(value) for value in values[:-1]])
            labels.append(int(values[-1]))
        return np.array(data), np.array(labels)
    
def plot_data_and_separator(data, labels, weights, bias):
    plt.figure(figsize=(8, 6))
    for i in range(len(data)):
        if labels[i] == 1:
            plt.scatter(data[i][0], data[i][1], color='blue', marker='o')
        else:
            plt.scatter(data[i][0], data[i][1], color='red', marker='x')

    x_values = np.linspace(min(data[:,0]), max(data[:,0]), 100)
    y_values = (-weights[0] / weights[1]) * x_values - (bias / weights[1])
    plt.plot(x_values, y_values, color='green')

    plt.title('Data and Separator Line')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.show()

def main():
    # Lectura de parámetros de usuario
    file_path = "OR_trn.csv"
    max_epochs = int(input("Introduce el número máximo de épocas de entrenamiento: "))
    learning_rate = float(input("Introduce la tasa de aprendizaje: "))

    # Lectura de datos de entrenamiento
    training_inputs, labels = read_training_data(file_path)
    num_inputs = training_inputs.shape[1]

    # Creación y entrenamiento del perceptrón
    perceptron = Perceptron(num_inputs, learning_rate)
    perceptron.train(training_inputs, labels, max_epochs)

    # Prueba del perceptrón con nuevos datos
    while True:
        new_inputs = input("Introduce las nuevas entradas separadas por comas (o 'salir' para salir): ")
        if new_inputs.lower() == 'salir':
            break
        new_inputs = [float(value) for value in new_inputs.split(',')]
        prediction = perceptron.predict(new_inputs)
        print("Predicción:", prediction)
    
    plot_data_and_separator(training_inputs, labels, perceptron.weights, perceptron.bias)

main()

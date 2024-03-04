import numpy as np
import random

class Perceptron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += learning_rate * (label - prediction) * inputs
                self.bias += learning_rate * (label - prediction)

def load_data(filename):
    dataset = np.loadtxt(filename, delimiter=',')
    inputs = dataset[:, :-1]
    labels = dataset[:, -1]
    return inputs, labels

def generate_partitions(inputs, labels, num_partitions, train_percentage):
    partitions = []
    data = list(zip(inputs, labels))
    random.shuffle(data)
    partition_size = len(data) // num_partitions
    for i in range(num_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size if i < num_partitions - 1 else len(data)
        partition = data[start:end]
        train_size = int(train_percentage * len(partition))
        train_set = partition[:train_size]
        test_set = partition[train_size:]
        train_inputs, train_labels = zip(*train_set)
        test_inputs, test_labels = zip(*test_set)
        partitions.append((np.array(train_inputs), np.array(train_labels), np.array(test_inputs), np.array(test_labels)))
    return partitions

def main():
    filename = "spheres2d70.csv"
    inputs, labels = load_data(filename)
    num_partitions = int(input("Introduce el número de particiones: "))
    train_percentage = float(input("Introduce el porcentaje de patrones de entrenamiento (0-1): "))
    partitions = generate_partitions(inputs, labels, num_partitions, train_percentage)
    accuracy_sum = 0
    for i, partition in enumerate(partitions):
        print(f"\nPartición {i+1}:")
        train_inputs, train_labels, test_inputs, test_labels = partition
        perceptron = Perceptron(num_inputs=train_inputs.shape[1])
        perceptron.train(train_inputs, train_labels)
        correct = 0
        for inputs, label in zip(test_inputs, test_labels):
            prediction = perceptron.predict(inputs)
            if prediction == label:
                correct += 1
        accuracy = correct / len(test_labels)
        accuracy_sum += accuracy
        print(f"Exactitud: {accuracy:.2f}")

    average_accuracy = accuracy_sum / num_partitions
    print(f"\nExactitud promedio en todas las particiones: {average_accuracy:.2f}")

if __name__ == "__main__":
    main()

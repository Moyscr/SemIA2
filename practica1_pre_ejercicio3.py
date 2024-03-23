import numpy as np
import matplotlib.pyplot as plt

# Definir la función a optimizar
def func(x, y):
    return 10 - np.exp(-(x**2 + 3*y**2))

# Calcular el gradiente de la función
def grad_func(x, y):
    df_dx = 2 * x * np.exp(-(x**2 + 3*y**2))
    df_dy = 6 * y * np.exp(-(x**2 + 3*y**2))
    return np.array([df_dx, df_dy])

# Algoritmo de descenso de gradiente
def gradient_descent(lr, num_iterations):
    # Inicializar valores aleatorios dentro del rango [-1, 1]
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    errors = []
    
    for _ in range(num_iterations):
        gradient = grad_func(x, y)
        x -= lr * gradient[0]
        y -= lr * gradient[1]
        # Limitar los valores dentro del rango [-1, 1]
        x = np.clip(x, -1, 1)
        y = np.clip(y, -1, 1)
        
        # Calcular el error y agregarlo a la lista de errores
        error = func(x, y)
        errors.append(error)
        
    return x, y, func(x, y), errors

# Parámetros
learning_rate = 0.1
num_iterations = 1000

# Ejecutar el algoritmo de descenso de gradiente
x_opt, y_opt, optimal_value, errors = gradient_descent(learning_rate, num_iterations)

# Imprimir resultados
print("Valor óptimo de x:", x_opt)
print("Valor óptimo de y:", y_opt)

# Graficar convergencia del error
plt.plot(errors)
plt.title('Convergencia del Error')
plt.xlabel('Iteración')
plt.ylabel('Valor de la función')
plt.show()

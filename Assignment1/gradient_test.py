import numpy as np
import matplotlib.pyplot as plt


def gradient_test(function, grad_function, X, W, C, iterations, epsilon=0.1):
    # Initialize epsilon and arrays to store errors
    zero_order_errors = []
    first_order_errors = []
    epsilons = [epsilon * (0.5 ** i) for i in range(iterations)]

    # Calculate the function value and gradient at x
    F0 = function(X, W, C)
    gradF, gradG = grad_function(X, W, C)

    dW = np.random.randn(*W.shape)
    dW /= np.linalg.norm(dW)  # Normalize to ensure consistent scaling

    for eps in epsilons:
        W_eps = W + eps * dW
        F1 = function(X, W_eps, C)

        zero_order_error = abs(F1 - F0)
        zero_order_errors.append(zero_order_error)

        # print(gradF_W.shape, dW.shape)
        first_order_error = np.abs(F1 - F0 - eps * np.dot(gradF.flat, dW.flat))
        first_order_errors.append(first_order_error)

    return zero_order_errors, first_order_errors


def run_gradient_test(function, grad_function, epsilon=1e-4, iterations=20):
    W = np.random.rand(4, 3) #nXl
    X = np.random.rand(4, 2) #nXm
    C = np.array([[1,0,0],
                  [0,0,1]]) #mXl

    errors_zero_order, errors_first_order = gradient_test(function, grad_function, X, W, C, iterations)
    plt.figure(figsize=(iterations, 6))
    plt.semilogy(range(iterations), errors_zero_order, label='Zero-order error', marker='o')
    plt.semilogy(range(iterations), errors_first_order, label='First-order error', marker='x')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Loss function gradient test')
    plt.legend()
    plt.show()


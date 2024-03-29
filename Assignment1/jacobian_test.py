import numpy as np
import matplotlib.pyplot as plt


def jacobian_test(function, jacMV, x, epsilon=1e-4, iterations=10):
    zero_order_errors = []
    first_order_errors = []
    epsilons = [epsilon * (0.5 ** i) for i in range(iterations)]
    # d = np.random.randn(*x.shape)

    for eps in epsilons:
        # Random direction vector d, scaled by eps
        d = np.random.randn(*x.shape)
        v = eps * d  # v = epsilon * d

        # Compute the function value at x and at x + epsilon * d
        f_x = function(x)
        f_x_eps = function(x + v)

        # Zero-order error: ||f(x + epsilon * d) - f(x)||
        zero_order_error = np.linalg.norm(f_x_eps - f_x)
        zero_order_errors.append(zero_order_error)

        # First-order error: ||f(x + epsilon * d) - f(x) - JacMV(x, epsilon * d)||
        jacMV_value = jacMV(x, v)
        first_order_error = np.linalg.norm(f_x_eps - f_x - jacMV_value)
        first_order_errors.append(first_order_error)

    return zero_order_errors, first_order_errors


def main():
    # Define the function and Jacobian-vector product
    def tanh(x):
        return np.tanh(x)

    def JacMV_tanh(x, v):
        # Jacobian-vector product for tanh is (1 - tanh^2(x)) * v
        return (1 - np.tanh(x) ** 2) * v

    # Perform the Jacobian check
    x = np.random.randn(20)
    zero_order_errors, first_order_errors = jacobian_test(tanh, JacMV_tanh, x, epsilon=1e-4, iterations=10)
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(10), zero_order_errors, label='Zero-order error', marker='o')
    plt.semilogy(range(10), first_order_errors, label='First-order error', marker='x')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Jacobian Test')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()



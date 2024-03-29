import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import random

from Assignment1.softmax import soft_max


def least_squares(A, x, b):
    return np.linalg.norm(A.dot(x) - b) ** 2 / 2


def least_squares_grad(A, x, b):
    return A.T.dot(A.dot(x) - b)


def calculate_accuracy(W, b, X, Y):
    y_pred = np.argmax(soft_max(X, W, b), axis=0).reshape(Y.shape[0], 1)
    return np.sum(y_pred == Y) / Y.shape[0]


def predict(X, W, b):
    probabilities = soft_max(X, W, b)
    # m*2 to m*1
    return np.argmax(probabilities, axis=0)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# W is the weight matrix
# X is the input matrix
# Y is the output matrix

def sgd(lossFunction, gradFunction, W, X_train, Y_train, X_test, Y_test, n_epochs, learning_rate, batch_size, b=0):
    loss = []
    train_accuracies = []
    valid_accuracies = []
    for epoch in range(n_epochs):
        # Decrease the learning rate every 100 epochs to
        if epoch % 100 == 0:
            learning_rate /= 10
        # Shuffle the data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        # Run the batches
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]
            grad_w, _ = gradFunction(X_batch, W, Y_batch, b)  # Unpack the two gradients
            W = W - learning_rate * grad_w

        # Calculate training accuracy
        train_pred = predict(X_train, W, b)
        train_acc = accuracy(np.argmax(Y_train, axis=0), train_pred)
        train_accuracies.append(train_acc)
        # Calculate validation accuracy
        valid_pred = predict(X_test, W, b)
        valid_acc = accuracy(np.argmax(Y_test, axis=0), valid_pred)
        valid_accuracies.append(valid_acc)
        # Record the average loss per epoch
        loss_epoch = lossFunction(X_train, W, Y_train, b)
        loss.append(loss_epoch)

    return W, loss, train_accuracies, valid_accuracies


def main():
    # Generate random data
    A = random(1000, 1000, density=0.01, format='csr')
    x = np.random.randn(1000)
    b = A.dot(x)

    # Run SGD
    W = np.random.randn(1000)
    W, loss, accuracy_train = sgd(least_squares, least_squares_grad, W, A, b, 100, 0.1, 100)

    # Plot the loss
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    main()

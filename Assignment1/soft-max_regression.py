import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def predict(X, W, b):
    probabilities = soft_max(X, W, b)
    return np.argmax(probabilities, axis=1)  # axis=1 for column-wise operation


def accuracy(y_true, y_pred):
    # If y_true is one-hot encoded, convert it to label indices
    y_true_labels = np.argmax(y_true, axis=0)
    return np.sum(y_pred == y_true_labels) / y_true_labels.shape[0]


def loss_function(X, W, C, b):
    m = X.shape[1]
    softmax = soft_max(X, W, b)
    return - np.sum(C.T * np.log(softmax)) / m


def loss_function_deriv_b(X, W, C, b):
    m = X.shape[1]
    return np.sum((soft_max(X, W, b) - C.T).T, axis=1) / m


def loss_function_deriv_w(X, W, C, b):
    m = X.shape[1]
    return (X @ (soft_max(X, W, b) - C.T)).T / m  # need to add .T


def loss_function_grad(X, W, C, b):
    return [loss_function_deriv_w(X, W, C, b), loss_function_deriv_b(X, W, C, b)]


def soft_max(X, W, b):
    linear_model_output = X.T @ W + b
    linear_model_output -= np.max(linear_model_output, axis=1, keepdims=True)
    exp_logits = np.exp(linear_model_output)
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return softmax


# W is the weight matrix
# X is the input matrix
# Y is the output matrix

def sgd(W, b, x_train, y_train, n_epochs, learning_rate, batch_size, x_test, y_test):
    loss = []
    train_accuracies = []
    valid_accuracies = []
    for epoch in range(n_epochs):
        # Decrease the learning rate every 100 epochs to
        if epoch % 100 == 0:
            learning_rate /= 10
        # Shuffle the data
        indices = np.arange(x_train.shape[1])  # use the second dimension for creating indices
        np.random.shuffle(indices)  # shuffling the columns while keeping the relationship between features and labels intact is crucial for SGD
        X_shuffled = x_train[:, indices]
        Y_shuffled = y_train[:, indices]

        # Run the batches
        for i in range(0, x_train.shape[1], batch_size):
            X_batch = X_shuffled[:, i:i + batch_size]  # (2, batch_size)
            Y_batch = Y_shuffled[:, i:i + batch_size]  # (2, batch_size)
            grad_w = loss_function_deriv_w(X_batch, W, Y_batch, b)  # Unpack the two gradients
            grad_b = loss_function_deriv_b(X_batch, W, Y_batch, b)
            W = W - learning_rate * grad_w.T / batch_size  # Update the weights ensures that the scale of the weight updates does not directly depend on the batch size.
            b = b - learning_rate * grad_b.T / batch_size

        # Calculate training accuracy
        train_pred = predict(x_train, W, b)
        train_acc = accuracy(y_train, train_pred)
        train_accuracies.append(train_acc)
        # Calculate validation accuracy
        valid_pred = predict(x_test, W, b)
        valid_acc = accuracy(y_test, valid_pred)
        valid_accuracies.append(valid_acc)
        # Record the average loss per epoch
        loss_epoch = loss_function(x_train, W, y_train, b)
        loss.append(loss_epoch)

    return W, loss, train_accuracies, valid_accuracies


# the matrices Yt,Ct represent the training data and Yv,Cv represent
# the validation data. The matrix Yt is the matrix of the input data


def main():
    data = scipy.io.loadmat('Data/GMMData.mat')

    # load the data
    x_train = data['Yt']
    y_train = data['Ct']
    x_test = data['Yv']
    y_test = data['Cv']

    w = np.random.randn(5, 5)
    w = w / np.linalg.norm(w)
    b = np.random.rand(1, 5)
    b = b / np.linalg.norm(b)
    n_epochs = 600
    learning_rate = 1
    batch_size = 100

    # Modify sgd to return validation accuracy as well
    Weights, loss_history, train_accuracies, valid_accuracies = sgd(w, b, x_train, y_train, n_epochs, learning_rate,
                                                                    batch_size, x_test, y_test)

    # Plot the training and validation accuracy
    plt.figure()
    plt.title(f'SGD - Soft max loss function, lr = {learning_rate}, batch size = {batch_size}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')

    plot1, = plt.plot(loss_history)

    plt.legend([plot1, ], ['F(W)'])
    plt.show()

    plt.figure()
    plt.title(f'SGD Accuracy, lr = {learning_rate}, batch size = {batch_size}')
    plt.xlabel('Epochs')
    plt.ylabel('Success Rate')

    plot1, = plt.plot(train_accuracies)
    plot2, = plt.plot(valid_accuracies)

    plt.legend([plot1, plot2], ['Train', 'Test'])
    plt.show()


if __name__ == '__main__':
    main()

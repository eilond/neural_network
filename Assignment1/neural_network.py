import numpy as np
import scipy
from matplotlib import pyplot as plt
from softmax import loss_function, loss_function_grad


class NeuralNetwork:
    def __init__(self, layers_dims, X_train, Y_train, X_test, Y_test, activation_function, activation_function_derivative):
        # Initialize weights and biases
        self.W = [np.random.randn(layers_dims[i], layers_dims[i - 1]) for i in range(1, len(layers_dims))]
        # self.W = [self.W[i]/np.linalg.norm(self.W[i]) for i in range(len(self.W))]
        self.b = [np.random.randn(layers_dims[i], 1) for i in range(1, len(layers_dims))]
        # self.b = [self.b[i] / np.linalg.norm(self.b[i]) for i in range(len(self.b))]
        self.architecture = layers_dims
        # self.W = [np.random.randn(layers_dims[i], layers_dims[i + 1]) for i in range(0, len(layers_dims)-1)]
        # self.b = [np.random.randn(layers_dims[i], 1) for i in range(0, len(layers_dims)-1)]
        self.X_test = X_test
        self.Y_test = Y_test
        self.new_W = self.W
        self.new_b = self.b
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.X_train = X_train
        self.Y_train = Y_train
        self.C_train = Y_train.T

        # print(f"layer dims:{layers_dims}")
        # print("this is shape b")
        # for bi in self.b:
        #     print(bi.shape)
        # print("this is shape W")
        # for wi in self.W:
        #     print(wi.shape)

    def update_weights(self, grad_W, grad_b, learning_rate):
        # print(f" this is grad_b len:{len(grad_b)}\n this is b len:{len(self.b)}\n this is grad_W len:{len(grad_W)}\n this is W len:{len(self.W)}")
        # for i in range(len(grad_b)):
        #     print(f"this is grad_b[i] shape:{grad_b[i].shape}")
        # for i in range(len(self.b)):
        #     print(f"this is b[i] shape:{self.b[i].shape}")
        self.W = [self.W[i] - learning_rate * grad_W[i] for i in range(len(grad_W))]
        self.b = [self.b[i] - learning_rate * grad_b[i].reshape(self.b[i].shape) for i in range(len(grad_b))]



    def test(self, X, Y):
        # print(Y)
        pred = self.forward_pass(X, test_mode=False)[-1]
        predicted_labels = np.argmax(pred, axis=0)
        correct_predictions = np.equal(predicted_labels, np.argmax(Y, axis=0))
        # print(correct_predictions)
        return np.mean(correct_predictions)

    def forward_pass(self, X, test_mode):
        # Forward propagation
        # print(X.shape)
        if test_mode:
            W = self.new_W
            b = self.new_b
        else:
            W = self.W
            b = self.b

        # print(f"this is b: {b}")
        X_acc = [X]
        # print(X_acc[0].shape)
        #hidden layers
        for i in range(len(W) - 1):
            # print(f"forwarding hidden layer num:{i}")
            X_acc.append(self.activation_function(np.dot(W[i], X_acc[i]) + b[i]))
        #softmax classifier
        X_acc.append(self.soft_max_classify_last_layer(X_acc[-1], W[-1], b[-1]))



        # print(X_acc[0].shape)
        # print(f"the len returned from the forward pass is:{len(X_acc)}")
        return X_acc

    def backward_pass(self, X, C):
        F_gradW = []
        F_gradb = []

        # for x in updated_X:
        #     print(x.shape)
        # for i in range(len(X)):
        #     print(f"given to backPass X is:{X[i].shape}")

        gradW, gradX = loss_function_grad(X[-2], self.W[-1], C)
        F_gradW.append(gradW)
        F_gradb.append(np.zeros_like(self.b[-1]))
        new_v = gradX
        # print(f"the len of X is:{len(X)}")
        # print(f"w len is:{len(self.W)}")
        for i in range(len(self.W) - 2, -1, -1):
            # print(f"num iter : {i}")
            curr_gradW, curr_gradb, new_v = self.grad_per_layer(self.W[i], X[i], self.b[i], new_v, self.activation_function_derivative)
            # print(f"grad b is: {curr_gradb}")
            # print(f"grad W is: {curr_gradW}")
            F_gradW.append(curr_gradW)
            F_gradb.append(curr_gradb)
        F_gradW.reverse()
        F_gradb.reverse()
        return F_gradW, F_gradb

    def grad_per_layer(self, W, X, b, v, activation_grad):
        act_grad_on_WXB_T_v = activation_grad(np.dot(W, X) + b) * v
        grad_w = np.dot(act_grad_on_WXB_T_v, X.T)
        grad_b = np.sum(act_grad_on_WXB_T_v, axis=1)
        new_v = np.dot(W.T, act_grad_on_WXB_T_v)
        return grad_w, grad_b, new_v

    def soft_max_classify_last_layer(self, X, W, b):
        linear_model_output = np.dot(W, X) + b
        linear_model_output -= np.max(linear_model_output, axis=0, keepdims=True)
        exp_logits = np.exp(linear_model_output)
        softmax_prob_classify = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        return softmax_prob_classify

    def gradient_test_full_network(self, epsilon=0.0001, iterations=10):
        zero_order_errors = []
        first_order_errors = []
        epsilons = [epsilon * (0.5 ** i) for i in range(iterations)]

        acc_X_in_each_layer = self.forward_pass(self.X_train, test_mode=True)
        Gw, Gb = self.backward_pass(acc_X_in_each_layer, self.C_train)
        # print(f"Jw:\n {Jw} \n Jb: {Jb}")
        F0 = loss_function(acc_X_in_each_layer[-2], self.W[-1], self.C_train)

        dW = [np.random.rand(self.W[i].shape[0], self.W[i].shape[1]) for i in range(len(self.W))]
        dW = [dW[i] / np.linalg.norm(dW[i]) for i in range(len(dW))]
        db = [np.random.rand(self.b[i].shape[0], self.b[i].shape[1]) for i in range(len(self.b))]
        db = [db[i] / np.linalg.norm(db[i]) for i in range(len(db))]

        for eps in epsilons:
            self.new_W = [self.W[i] + eps * dW[i] for i in range(len(dW))]
            self.new_b = [self.b[i] + eps * db[i] for i in range(len(db))]

            X_acc_plus_db_dw = self.forward_pass(self.X_train, test_mode=True)

            F1 = loss_function(X_acc_plus_db_dw[-2], self.new_W[-1], self.C_train)
            zero_order_errors.append(abs(F1 - F0))
            first_order_errors.append(abs(F1 - F0 - eps * sum(np.sum(dW[i] * Gw[i]) for i in range(len(self.new_W)))
                                          - eps * sum(np.sum(db[i].flat * Gb[i]) for i in range(len(self.new_b)))))

        plt.figure()
        plt.semilogy(zero_order_errors, label="Zero-order error", marker="o")
        plt.semilogy(first_order_errors, label="First-order error", marker="x")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title("Gradient Test full network")
        plt.legend()
        plt.show()

    def jac_test_hidden_layer(self, epsilon=0.1, iterations=10):
        zero_order_errors = []
        first_order_errors = []
        epsilons = [epsilon * (0.5 ** i) for i in range(iterations)]

        X = np.random.randn(2, 1)
        w = np.random.randn(2, 2)
        b = np.random.randn(2, 1)

        dv = np.random.randn(2, 1)
        dv /= np.linalg.norm(dv)
        dW = np.random.randn(2, 2)
        dW /= np.linalg.norm(dW)
        db = np.random.randn(2, 1)
        db /= np.linalg.norm(db)

        # Jacobian transpose times a random vector V to test if the implementation
        # of the formulas shown in class is correct
        F0 = np.dot(self.activation_function(np.dot(w, X) + b).T, dv)
        Jw, Jb, Jx = self.grad_per_layer(w, X, b, dv, self.activation_function_derivative)

        for eps in epsilons:
            w_eps_d = w + eps * dW
            b_eps_d = b + eps * db
            F1 = np.dot(self.activation_function(np.dot(w_eps_d, X) + b_eps_d).T, dv)
            result = abs(F1 - F0).flatten()
            testResult = abs(
                F1 - F0 - eps * np.dot(dW.flatten(), Jw.flatten()) - eps * np.dot(db.flatten(), Jb.flatten())).flatten()
            zero_order_errors.append(result)
            first_order_errors.append(testResult)

        plt.figure()
        plt.semilogy(zero_order_errors, label="Zero-order error", marker="o")
        plt.semilogy(first_order_errors, label="First-order error", marker="x")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title("Hidden layer Jacobian test")
        plt.legend()
        plt.show()

    def train(self, n_epochs, learning_rate, batch_size):
        original_learning_rate = learning_rate
        loss = []
        train_accuracies = []
        test_accuracy = []
        for epoch in range(n_epochs):
            # Decrease the learning rate every 100 epochs to
            if epoch % 100 == 0:
                learning_rate /= 10
            # Shuffle the data
            indices = np.arange(self.X_train.shape[1])  # use the second dimension for creating indices
            np.random.shuffle(indices)  # shuffling the columns while keeping the relationship between features and labels intact is crucial for SGD
            X_shuffled = self.X_train[:, indices]
            Y_shuffled = self.Y_train[:, indices]

            # Split the data into batches
            for i in range(0, self.X_train.shape[1], batch_size):
                X_batch = X_shuffled[:, i:i + batch_size]  # (2, batch_size)
                Y_batch = Y_shuffled[:, i:i + batch_size]  # (2, batch_size)
                # C_batch = self.get_C_mat(Y_batch)
                C_batch = Y_batch.T

                # Forward pass
                acc_X_in_each_layer = self.forward_pass(X_batch, test_mode=False)
                # Backward pass
                grad_W, grad_b = self.backward_pass(acc_X_in_each_layer, C_batch)

                # Update weights and biases
                self.update_weights(grad_W, grad_b, learning_rate)

            acc_X_in_each_layer = self.forward_pass(self.X_train, test_mode=False)
            loss.append(loss_function(acc_X_in_each_layer[-2], self.W[-1], self.C_train))
            train_accuracies.append(self.test(self.X_train, self.Y_train)*100)
            test_accuracy.append(self.test(self.X_test, self.Y_test)*100)

        self.plot(loss, train_accuracies, test_accuracy, original_learning_rate, batch_size)

    def plot(self, loss, train_accuracies, test_accuracies,learning_rate, batch_size):
        plt.figure()
        plt.title(f"Network loss.")
        plt.xlabel("Epochs")
        plt.ylabel("Loss value")
        plt.plot(loss, label="Loss function")
        plt.legend()
        plt.show()

        plt.figure()
        plt.title(f"Network accuracy (hit percentage).")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Test accuracy")
        plt.legend()
        plt.show()


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def run_test_SwissRollData(file_path):
    np.random.seed(36)  # For reproducibility
    data = scipy.io.loadmat(file_path)
    x_train = data.get("Yt")
    # x_train = data.get("Yt")[:,:200]
    y_train = (data.get("Ct"))
    # y_train = (data.get("Ct"))[:,:200]
    x_test = data.get("Yv")
    y_test = (data.get("Cv"))
    sample_dims = x_train.shape[0]
    hidden_layer_size = [2, 5, 7, 11, 5, 2]
    output_layer_size = y_train.shape[0]
    nn = NeuralNetwork(layers_dims=[sample_dims, *hidden_layer_size, output_layer_size],
                       X_train=x_train,
                       Y_train=y_train,
                       X_test=x_test,
                       Y_test=y_test,
                       activation_function=tanh,
                       activation_function_derivative=tanh_derivative)


    # nn.jac_test_hidden_layer()
    # nn.gradient_test_full_network()
    nn.train(200, 0.1, 50)




def run_test_GMMData(file_path):
        np.random.seed(2)
        # np.random.seed(15)
        data = scipy.io.loadmat(file_path)
        # x_train = data.get("Yt")
        x_train = data.get("Yt")[:,:200]
        # y_train = (data.get("Ct"))
        y_train = (data.get("Ct"))[:,:200]
        x_test = data.get("Yv")
        y_test = (data.get("Cv"))
        sample_dims = x_train.shape[0]
        hidden_layer_size = [5, 7, 11, 15, 11, 5, 5]
        # hidden_layer_size = [2, 5, 7, 10,10, 10,5, 5, 5]
        output_layer_size = y_train.shape[0]
        nn = NeuralNetwork(layers_dims=[sample_dims, *hidden_layer_size, output_layer_size],
                           X_train=x_train,
                           Y_train=y_train,
                           X_test=x_test,
                           Y_test=y_test,
                           activation_function=relu,
                           activation_function_derivative=relu_derivative)

        nn.train(200, 0.1, 50)


def main():
    # run_test_SwissRollData("Data/SwissRollData.mat")
    run_test_GMMData("Data/GMMData.mat")


# best results!! nn.train(300, 0.1, 20)
# nn.train(200, 0.1, 50)
# nn.train(300, 0.1, 50)

if __name__ == "__main__":
    main()
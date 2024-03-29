import numpy as np
from gradient_test import run_gradient_test

# X matrix with x_i as column vector
# W matrix with w_i as column vector
def soft_max(X, W):
    linear_model_output = np.dot(X.T, W)
    linear_model_output -= np.max(linear_model_output, axis=1, keepdims=True)
    exp_logits = np.exp(linear_model_output)
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return softmax


def loss_function(X, W, C):
    m = X.shape[1]
    softmax = soft_max(X, W)
    ret = - np.sum(C * np.log(softmax)) / m
    # print(f"loss function returned: {ret}")
    return ret


def loss_function_grad(X, W, C):
    m = X.shape[1]
    # print(f" W shape: {W.shape} \n X shape: {X.shape} \n C shape: {C.shape}")
    grad_W = (np.dot(X, (soft_max(X, W) - C))) / m
    grad_X = (np.dot(W, (soft_max(X, W) - C).T)) / m
    return grad_W, grad_X



if __name__ == "__main__":
    run_gradient_test(loss_function, loss_function_grad)

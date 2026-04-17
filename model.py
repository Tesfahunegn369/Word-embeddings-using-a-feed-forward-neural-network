import numpy as np
from utils2 import get_dict, get_batches, softmax

## Q2: Forward Propagation (15 pts)
def forward_prop(x, W1, W2, b1, b2):
    '''
    Inputs:
        x:  average one hot vector for the context
        W1, W2, b1, b2:  matrices and biases to be learned
     Outputs:
        z:  output score vector
    '''

    ### START CODE HERE (Replace instances of 'None' with your own code) ###

    # Calculate h
    h = np.dot(W1, x) + b1

    # Apply the relu on h (store result in h)
    h = np.maximum(0, h)

    # Calculate z
    z = np.dot(W2, h) + b2

    ### END CODE HERE ###

    return z, h


## Q4: Back Propagation (25 pts)
def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    '''
    Inputs:
        x:  average one hot vector for the context
        yhat: prediction (estimate of y)
        y:  target vector
        h:  hidden vector (see eq. 1)
        W1, W2, b1, b2:  matrices and biases
        batch_size: batch size
     Outputs:
        grad_W1, grad_W2, grad_b1, grad_b2:  gradients of matrices and biases
    '''
    ### START CODE HERE (Replace instanes of 'None' with your code) ###

    # Compute l1 as W2^T (Yhat - Y)
    # Re-use it whenever you see W2^T (Yhat - Y) used to compute a gradient
    l1 = np.dot(W2.T, (yhat - y)) 
    # Apply relu to l1
    l1 = l1 * (h > 0)
    # Compute the gradient of W1
    grad_W1 = np.dot(l1, x.T) / batch_size
    # Compute the gradient of W2
    grad_W2 = np.dot((yhat - y), h.T) / batch_size
    # Compute the gradient of b1
    grad_b1 = np.sum(l1, axis=1, keepdims=True) / batch_size
    # Compute the gradient of b2
    grad_b2 = np.sum((yhat - y), axis=1, keepdims=True) / batch_size
    ### END CODE HERE ###

    return grad_W1, grad_W2, grad_b1, grad_b2


################################################################################


def forward_prop_test():
    from training_util import initialize_model
    # Test the function
    print("\n## Q2: Forward Propagation (15 pts)")

    # Create some inputs
    tmp_N = 5
    tmp_V = 3
    tmp_x = np.array([[0, 1, 0]]).T
    tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(N=tmp_N, V=tmp_V, random_seed=1)

    print(f"x has shape {tmp_x.shape}")
    print(f"Dimension of hidden vectors is {tmp_N} and vocabulary size V is {tmp_V}")

    # call function
    print("call forward_prop")
    print()
    tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)

    # Check output
    print(f"z has shape {tmp_z.shape}")
    if np.abs(tmp_z - np.array([[[-0.0674], [-0.2708], [ 0.0115]]])).mean() < 0.001:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)
    print()

    # Check hidden vector
    print(f"h has shape {tmp_h.shape}")
    if np.abs(tmp_h - np.array([[0.], [0.4686], [0.], [0.6169], [0.]])).mean() < 0.001:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)
    print()


def back_prop_test(data):
    from training_util import initialize_model
    # Test the function
    print("\n## Q4: Back Propagation (25 pts)")

    tmp_C = 2
    tmp_N = 50
    tmp_batch_size = 4
    tmp_word2Ind, tmp_Ind2word = get_dict(data)
    tmp_V = len(tmp_word2Ind)

    # get a batch of data
    tmp_x, tmp_y = next(get_batches(data, tmp_word2Ind, tmp_V, tmp_C, tmp_batch_size))

    print("get a batch of data")
    print(f"tmp_x.shape {tmp_x.shape}")
    print(f"tmp_y.shape {tmp_y.shape}")

    print()
    print("Initialize weights and biases")
    tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N, tmp_V)

    print(f"tmp_W1.shape {tmp_W1.shape}")
    print(f"tmp_W2.shape {tmp_W2.shape}")
    print(f"tmp_b1.shape {tmp_b1.shape}")
    print(f"tmp_b2.shape {tmp_b2.shape}")

    print()
    print("Forwad prop to get z and h")
    tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
    print(f"tmp_z.shape: {tmp_z.shape}")
    print(f"tmp_h.shape: {tmp_h.shape}")

    print()
    print("Get yhat by calling softmax")
    tmp_yhat = softmax(tmp_z)
    print(f"tmp_yhat.shape: {tmp_yhat.shape}")

    print()
    print("call back_prop")
    tmp_m = (2 * tmp_C)
    tmp_grad_W1, tmp_grad_W2, tmp_grad_b1, tmp_grad_b2 = back_prop(tmp_x, tmp_yhat, tmp_y, tmp_h, tmp_W1, tmp_W2,
                                                                   tmp_b1, tmp_b2, tmp_batch_size)

    print(f"tmp_grad_W1.shape {tmp_grad_W1.shape}")
    print(f"tmp_grad_W2.shape {tmp_grad_W2.shape}")
    print(f"tmp_grad_b1.shape {tmp_grad_b1.shape}")
    print(f"tmp_grad_b2.shape {tmp_grad_b2.shape}")

    if np.abs(tmp_grad_W2[:3, :3] - np.array([[0., 0., 0.], [0., 0., -1.5313], [0., 0., 0.]])).mean() < 0.0001:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)

    if np.abs(tmp_grad_b1[:10, 0] - np.array([0., 0., 1.4415, 0., 0., 3.3070, 1.4642, 1.3024, 0., 0.])).mean() < 0.0001:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)
    print()


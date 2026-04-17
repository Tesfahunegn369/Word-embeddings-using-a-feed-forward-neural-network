import numpy as np
from utils2 import get_dict, get_batches, softmax
from model import forward_prop, back_prop


#Gradient Descent
def gradient_descent(data, word2Ind, N, V, C, num_iters, alpha=0.03):
    '''
    This is the gradient_descent function

      Inputs:
        data:      text
        word2Ind:  words to Indices
        N:         dimension of hidden vector
        V:         dimension of vocabulary
        C:         context window size
        num_iters: number of iterations
        alpha:     learning rate
     Outputs:
        W1, W2, b1, b2:  updated matrices and biases

    '''
    from training_util import initialize_model, compute_cost
    W1, W2, b1, b2 = initialize_model(N, V, random_seed=282)
    batch_size = 128
    iters = 0

    for x, y in get_batches(data, word2Ind, V, C, batch_size):
        # Get z and h
        z, h = forward_prop(x, W1, W2, b1, b2)
        # Get yhat
        yhat = softmax(z)
        # Get cost
        cost = compute_cost(y, yhat, batch_size)
        if ((iters + 1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")
        # Get gradients
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)

        # Update weights and biases
        W1 -= alpha * grad_W1
        W2 -= alpha * grad_W2
        b1 -= alpha * grad_b1
        b2 -= alpha * grad_b2

        iters += 1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66

    return W1, W2, b1, b2

def gradient_descent_test(data):
    # test your function
    print("\nGradient Descent")

    C = 2
    N = 50
    word2Ind, Ind2word = get_dict(data)
    V = len(word2Ind)
    num_iters = 150
    print("Call gradient_descent")
    W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, C, num_iters)  # The loss(cost) should decrease as the iteration progresses

    if np.abs(W1[:3, :3] - np.array([[-1.3185, -1.6022, -0.3668], [-1.7603, 0.9399, 1.9940], [-0.9512, -0.2248, -1.5313]])).mean() < 0.0001:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)

    if np.abs(b2[:10, 0] - np.array([-0.0513, 0.8844, -2.1104, -0.9547, -0.1285, -1.2403, 0.5311, 0.7425, -0.7103, -0.6085])).mean() < 0.0001:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)






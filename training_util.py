import numpy as np
from utils2 import get_batches, get_dict, softmax
from training import forward_prop

def initialize_model(N, V, random_seed=1):
    '''
    Inputs:
        N:  dimension of hidden vector
        V:  dimension of vocabulary
        random_seed: random seed for consistent results in the unit tests
     Outputs:
        W1, W2, b1, b2: initialized weights and biases
    '''
    np.random.seed(random_seed)

    # W1 has shape (N,V)
    W1 = np.random.normal(size=(N, V))
    # W2 has shape (V,N)
    W2 = np.random.normal(size=(V, N))
    # b1 has shape (N,1)
    b1 = np.random.normal(size=(N, 1))
    # b2 has shape (V,1)
    b2 = np.random.normal(size=(V, 1))

    return W1, W2, b1, b2


#Cost Function: Cross-entropy loss
def compute_cost(y, yhat, batch_size):
    '''
    Inputs:
        y:  truth labels. size: Vocab.size * batch_size
        yhat: predicted score for each word. size: Vocab.size * batch_size
        batch_size: batch size
     Outputs:
        cost:  average of cross-entropy loss for instances in the batch
    '''
    logprobs = np.sum(y * np.log(yhat), axis=0)

    # cost: -avg. logprobs
    cost = -np.sum(logprobs) / batch_size
    cost = np.squeeze(cost)
    return cost

def cost_func_test(data):
    # Test the function
    print("\nCost Function: Cross-entropy loss")

    tmp_C = 3
    tmp_N = 50
    tmp_batch_size = 4
    tmp_word2Ind, tmp_Ind2word = get_dict(data)
    tmp_V = len(tmp_word2Ind)

    tmp_x, tmp_y = next(get_batches(data, tmp_word2Ind, tmp_V, tmp_C, tmp_batch_size))

    print(f"tmp_x.shape {tmp_x.shape}")
    print(f"tmp_y.shape {tmp_y.shape}")

    tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N, tmp_V)

    print(f"tmp_W1.shape {tmp_W1.shape}")
    print(f"tmp_W2.shape {tmp_W2.shape}")
    print(f"tmp_b1.shape {tmp_b1.shape}")
    print(f"tmp_b2.shape {tmp_b2.shape}")

    tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
    print(f"tmp_z.shape: {tmp_z.shape}")
    print(f"tmp_h.shape: {tmp_h.shape}")

    tmp_yhat = softmax(tmp_z)
    print(f"tmp_yhat.shape: {tmp_yhat.shape}")

    print("call compute_cost")
    tmp_cost = compute_cost(tmp_y, tmp_yhat, tmp_batch_size)

    if tmp_cost < 20:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)
    print()


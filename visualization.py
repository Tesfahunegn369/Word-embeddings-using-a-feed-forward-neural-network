# visualizing the word vectors here
from matplotlib import pyplot
from utils2 import compute_pca

def visualization(W1, W2, word2Ind):
    words = ['king', 'queen', 'lord', 'man', 'woman', 'dog', 'horse',
             'rich', 'happy', 'sad']

    embs = (W1.T + W2) / 2.0

    # given a list of words and the embeddings, it returns a matrix with all the embeddings
    idx = [word2Ind[word] for word in words]
    X = embs[idx, :]
    print(X.shape, idx)  # X.shape:  Number of words of dimension N each

    result = compute_pca(X, 3)
    ax = pyplot.figure().add_subplot(projection='3d')
    ax.scatter(result[:, 0], result[:, 1], result[:, 2], s=20)

    for i, word in enumerate(words):
        ax.text(result[i, 0], result[i, 1], result[i, 2], word)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    pyplot.show()
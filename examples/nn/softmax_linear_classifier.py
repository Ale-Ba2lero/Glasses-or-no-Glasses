import numpy as np

N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
y = np.zeros(N * K, dtype='uint8')  # class labels
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j
# lets visualize the data:
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

# Normally we would want to preprocess the dataset 
# so that each feature has zero mean and unit standard deviation, 
# but in this case the features are already in a 
# nice range from -1 to 1, so we skip this step.

# Training a Softmax Linear Classifier
# initialize parameters randomly
W = 0.01 * np.random.randn(D, K)  # weights matrix
b = np.zeros((1, K))  # bias vector

# some hyperparameters
step_size = 1e-0
reg = 1e-3  # regularization strength

# Compute the loss -> Cross entropy (-log(prob))
num_examples = X.shape[0]

loss = 0

for i in range(10000):
    # evaluate class scores, [N x K], for a linear classifier
    scores = np.dot(X, W) + b

    # compute the class probabilities
    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalize them for each example
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    #reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss #+ reg_loss

    if i % 1000 == 0:
        print(f"iteration {i}: loss {loss}")

    # compute the gradient on scores w/ backpropagation
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg * W  # regularization gradient

    # Now that we’ve evaluated the gradient we know
    # how every parameter influences the loss function
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print(f'\ntraining accuracy: {(np.mean(predicted_class == y))}')

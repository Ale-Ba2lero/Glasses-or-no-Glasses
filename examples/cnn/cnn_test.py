# %%
from tqdm import tqdm
import idx2numpy
import numpy as np
from examples.cnn.cnn_layers import Conv3x3, MaxPool2, Softmax
import time

train_images_file = 'dataset/train-images.idx3-ubyte'
train_labels_file = 'dataset/train-labels.idx1-ubyte'
test_images_file = 'dataset/t10k-images.idx3-ubyte'
test_labels_file = 'dataset/t10k-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(train_images_file)
train_labels = idx2numpy.convert_from_file(train_labels_file)
test_images = idx2numpy.convert_from_file(test_images_file)
test_labels = idx2numpy.convert_from_file(test_labels_file)

train_images = train_images[:1000]
train_labels = train_labels[:1000]
test_images = test_images[:100]
test_labels = test_labels[:100]

conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10

global forward_time
global backward_time

forward_time = 0
backward_time = 0

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(img, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''

    start = time.time()
    # Forward
    out, loss, acc = forward(img, label)
    end = time.time()
    global forward_time
    forward_time += (end - start)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    start = time.time()
    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)
    end = time.time()
    global backward_time
    backward_time += (end - start)

    return loss, acc


print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in tqdm(range(3)):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train!
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

print(forward_time)
print(backward_time)
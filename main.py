from keras.datasets import mnist
from keras.utils import np_utils
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np


from network import Network
from c_layer import CLayer
from active_layer import ActiveLayer
from activation_functions import sigmoid, sigmoid_prime
from losses import mse, mse_prime

FILES = glob.glob ("C:/Users/aniem/Documents/Code/WSI/LAB5/samples/*.jpg")

def prep_data():
    # mnist data : 60000 samples
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape and normalize
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)

    # same for test data : 60000 samples
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)

def create_network(shape, activaion, activation_prime, loss, loss_prime, epochs=30, learning_rate=0.1, test_size=1000):
    (x_train, y_train), (x_test, y_test) = prep_data()
    net = Network()
    shape.insert(0, 28*28)
    shape.append(10)
    for previous, current in zip(shape, shape[1:]):
        net.add(CLayer(previous, current))
        net.add(ActiveLayer(activaion, activation_prime))

    # set loss function
    net.use(loss, loss_prime)
    # train
    net.fit(x_train[0:test_size], y_train[0:test_size], epochs, learning_rate)
    return net


def convert(path='samples', name=None):
    test_list = list()
    if not name:
        for file in FILES:
            image = cv2.imread (file, cv2.IMREAD_GRAYSCALE)
            image = image.reshape(1, 1, 28*28)
            image = image.astype('float32')
            image /= 255
            test_list.append(image)
        return test_list


def toDict(network):
    out = network.predict(convert())
    values = list()

    for list_ in out:
        dic = dict()
        enum = 0
        for prob in np.ndenumerate(list_):
            dic[enum] = prob[1]
            enum += 1
        values.append(dic)

    output = ""
    for enum, entry in enumerate(values):
        highest_val = max(entry, key=entry.get)
        output += (f"NETWORK CLASSIFIED {enum} as {highest_val} with probability of {entry[highest_val]}\n")

    return output


def main():
    network = create_network([100, 50, 25], sigmoid, sigmoid_prime, mse, mse_prime,test_size=30000, epochs=20)
    print(toDict(network))


if __name__ == '__main__':
    main()





# create_network([100,50], sigmoid, sigmoid_prime, mse, mse_prime)
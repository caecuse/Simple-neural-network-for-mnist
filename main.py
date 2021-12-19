import numpy as np
import cv2
import glob
import argparse
from matplotlib import pyplot as plt
from typing import Callable, List, Tuple, TypeVar
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
# from keras.datasets import mnist
# from keras.utils import np_utils
from network import Network
from c_layer import CLayer
from active_layer import ActiveLayer
from activation_functions import sigmoid, sigmoid_prime
from losses import mse, mse_prime
from os import getcwd
from tqdm import tqdm



A = TypeVar("A")
FILES = glob.glob(getcwd()+"/samples/*.jpg")


def prep_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    # mnist data : 60000 samples
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape and normalize
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = utils.to_categorical(y_train)

    # same for test data : 60000 samples
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = utils.to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)


def create_network(shape:List[int], x_data:np.ndarray, y_data:np.ndarray,
                   activation:Callable[[float, float], float],
                   activation_prime:Callable[[float, float], float],
                   loss:Callable[[float], float], loss_prime:Callable[[float], float],
                   epochs: int=30, learning_rate: float=0.1,
                   train_size: int=1000, print_info:bool=False) -> Network:
    net = Network()
    shape.insert(0, 28*28)
    shape.append(10)
    for previous, current in zip(shape, shape[1:]):
        net.add(CLayer(previous, current))
        net.add(ActiveLayer(activation, activation_prime))

    # set loss function
    net.use(loss, loss_prime)
    # train
    net.fit(x_data[0:train_size], y_data[0:train_size], epochs, learning_rate, print_info)
    return net


def convert() -> List[np.ndarray]:
    test_list = list()
    for file in FILES:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = image.reshape(1, 28*28)
        image = image.astype('float32')
        image = (255 - image) / 255
        test_list.append(image)
    return test_list


def test_real(network: Network) -> str:
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
        output += (f"NETWORK CLASSIFIED {enum} as {highest_val} " +
                   f"with probability of {entry[highest_val]}\n")

    return output


def test_mnist(network: Network, test_size: int, x_test: np.ndarray,
               y_test: np.ndarray) -> float:
    out = network.predict(x_test[:test_size])
    accuracy = 0
    for enum, elem in enumerate(out):
        max_index = np.argmax(elem[0])
        if y_test[enum][max_index] == 1:
            accuracy += 1
    return accuracy / test_size

def plotter(mnist_data: List[List[float]]):
    im = plt.imshow(mnist_data[::-1], extent=[0, len(mnist_data[0]), 0, len(mnist_data)])
    plt.colorbar(label="Accuracy of alg.")
    plt.xlabel("Epochs")
    plt.ylabel("Test size")
    plt.xticks(np.arange(0, len(mnist_data[0])+1, 1))
    plt.yticks(np.arange(0, len(mnist_data)+1, 1))
    plt.show()

def main(arguments: A) -> None:
    if not arguments['layers']:
        if arguments['train_size'] or arguments['epochs']:
            print("At least one layer has to be specified when train_size or epochs are provided")
        else:
            mnist_results = []
            train_size = [1000, 5000, 10000, 15000, 20000]
            epoch_size = [20, 40, 60, 80, 100]
            (x_train, y_train), (x_test, y_test) = prep_data()
            with tqdm(total=len(train_size)*len(epoch_size)) as progress_bar:
                for i, ts in (enumerate([1000, 5000, 10000, 15000, 20000])): #1000, 2000, 5000, 10000, 30000
                    mnist_results.append([])
                    for ep in (epoch_size):
                        network = create_network([200, 100, 50, 25], x_train, y_train, sigmoid, sigmoid_prime,
                                    mse, mse_prime, train_size=ts, epochs=ep)
                        mnist_results[i].append(test_mnist(network, ts, x_test, y_test))
                        progress_bar.update(1)
        plotter(mnist_results)
    else:
        ts = 1000
        ep = 10
        shape = arguments['layers']
        if arguments['train_size']:
            ts = arguments['train_size']
        if arguments['epochs']:
            ep = arguments['epochs']
        (x_train, y_train), (x_test, y_test) = prep_data()
        network = create_network(shape, x_train, y_train, sigmoid, sigmoid_prime,
                                mse, mse_prime, train_size=ts, epochs=ep, print_info=True)

        print("Accuracy for 1000 samples is:")
        print(test_mnist(network, 1000, x_test, y_test))

        print(test_real(network))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start up the neural network.')
    parser.add_argument('layers', metavar='layers', type=int, nargs='*')
    parser.add_argument('--train_size', type=int)
    parser.add_argument('--epochs', type=int)
    args = vars(parser.parse_args())
    main(args)

# python3 ./main.py 200 100 50 25 --train_size 30000  --epochs 10
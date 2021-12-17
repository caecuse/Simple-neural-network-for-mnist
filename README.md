
# Convolutional Neural Network

Kinda basic implemention of the most basic neural network you can make, specifically written for mnist dataset. Of course you can add as many layers as you want and add loss and activation functions yourself in proper files.

## Optimazing

Optimizing alghoritm for this implementation is the most simple gradient descent I came up with.

## Usage
To create network you need to add layers to `Network` instance. Since I divided implemention of layers into two smaller tasks you will need to add `CLayer` and then `ActivateLayer`. The easiest way to imagine those two classes is to think of `CLayer` as
a layer of nodes with random biases and weights and that `AcitveLayer` forces those values to change after some math.

So the way it works is:
```python
net = Network()
net.add(CLayer(28 * 28, nextLayerNeuronCount)) # You can guess what id does
net.add(ActiveLayer(activation, activation_prime)) # activation function & f'
net.add(CLayer(nextLayerNeuronCount, nexterLayerNeuronCount))
net.add(ActiveLayer(activation, activation_prime))
net.add(CLayer(nexterLayerNeuronCount, 10)) # last layer has 10 neurons
net.add(ActiveLayer(activation, activation_prime))
```
And thats neural network for ya. There absolutly more of them fancy parameteres in there like `learning_rate` and `epochs` for training that net, but who would bother setting them anyways.
# Accuracy
We cannot forget about tests now, can we? So in `main.py` you should find `test_mnist` and `test_real` functions. First one takes arguments from mnist database (sample to your liking) and checks them against the network and checks them hard - if the value is what it should be it's a hit, otherwise I don't care how close it was. If you like to, go ahead and play with it to give you what ya want. Second one on the other hand checks the samples I drew against the net, they are not perfect but I just like to see real-life usage of stuff like that irl so I did. If you want to drew samples yourself, go ahead just remember to not change the names in the file, cuz those names they the function (in a way) whether it's a hit or nay.
## Test
Paremetres of the test:
```py
network = create_network([200, 100, 50, 25], x_train, y_train, sigmoid, sigmoid_prime, mse, mse_prime, test_size=30000, epochs=10)
```
Which means that test was conducted on `Network` with 4 hidden layers, activation functions was the sigmoid one, to calculate loss mean squared error formula was used, size of the test was 30000 first entries from `mnist dataset` and we trained the set 10 times.
## Results
Overall accuracy for the `test_mnist` was `88.4%`, and output of `test_real` was as follows;
```
NETWORK CLASSIFIED 0 as 0 with probability of 0.9015740215283164
NETWORK CLASSIFIED 1 as 9 with probability of 0.7581752169782214
NETWORK CLASSIFIED 2 as 2 with probability of 0.9788834962324736
NETWORK CLASSIFIED 3 as 3 with probability of 0.5370101401752179
NETWORK CLASSIFIED 4 as 8 with probability of 0.23128159669599963
NETWORK CLASSIFIED 5 as 5 with probability of 0.8046032596204756
NETWORK CLASSIFIED 6 as 6 with probability of 0.40650633789413326
NETWORK CLASSIFIED 7 as 2 with probability of 0.778977682926675
NETWORK CLASSIFIED 8 as 8 with probability of 0.5401737087583314
NETWORK CLASSIFIED 9 as 3 with probability of 0.45197299505509914
```
Which is not too bad considering the fact we didnt use any droput technics nor regularization. If you want to pump up that score you can experiment with adding more layers to the `network` or just grabbing bigger `train_size` than I did.
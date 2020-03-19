# linear algebra and array functionality
import itertools

# plots & graphs
import matplotlib.pyplot as plt
import numpy as np

# data work
import pandas as pd
from pandas.tests.dtypes.test_inference import expected
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


# sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ReLU activation function
def re_lu(z):
    return np.maximum(0, z)


def d_re_lu(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


def d_sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    d_z = s * (1 - s)
    return d_z


# neuron network class
class DoubleLayerNetwork:
    def __init__(self, _input, _output):
        # input layer
        self.input = _input
        # output data
        self.output = _output
        # output data prediction
        self.predicted_output = np.zeros((1, self.output.shape[1]))
        # count of layers(excluding input layer)
        self.layers_count = 2
        # count of neurons for each level
        self.neurons = [3, 15, 1]
        # params W and b for each level
        self.param = {}
        # some cache
        self.cache = {}
        # loss value for each 500 iterations
        self.loss = []
        # learning rate
        self.lr = 0.003
        # number of training samples
        self.samples_count = self.output.shape[1]
        # threshold
        self.threshold = 0.5

    def init(self):
        # make random determinate
        np.random.seed(1)
        # weights for layer 1(hidden) is matrix [b * a]
        self.param['W1'] = np.random.randn(self.neurons[1], self.neurons[0]) / np.sqrt(self.neurons[0])
        # bias for layer 1(hidden) is vector [b * 1]
        self.param['b1'] = np.zeros((self.neurons[1], 1))
        # weights for layer 2(output) is matrix [c * b]
        self.param['W2'] = np.random.randn(self.neurons[2], self.neurons[1]) / np.sqrt(self.neurons[1])
        # bias for layer 2(output) is vector [c * 1]
        self.param['b2'] = np.zeros((self.neurons[2], 1))
        return

    @property
    def forward(self):
        # multiply input data by weights of layer 1(hidden), add bias and gets
        # intermediate value (output of hidden layer)
        z1 = self.param['W1'].dot(self.input) + self.param['b1']
        # apply activation function(ReLU) for hidden layer
        a1 = re_lu(z1)
        # save this in cache
        self.cache['Z1'], self.cache['A1'] = z1, a1

        # multiply intermediate value(A1) by weights of layer 2(output), add bias and gets
        # output value (prediction)
        z2 = self.param['W2'].dot(a1) + self.param['b2']
        # apply activation function(Sigmoid) for output layer
        a2 = sigmoid(z2)
        # save this in cache
        self.cache['Z2'], self.cache['A2'] = z2, a2
        # save prediction value
        self.predicted_output = a2
        # compute loss value
        loss_value = self.compute_loss(a2)

        return self.predicted_output, loss_value

    # compute loss value
    # We use Cross-Entropy Loss Function for classification problem
    def compute_loss(self, predicted_value):
        loss_value = (1. / self.samples_count) * (
                -np.dot(self.output, np.log(predicted_value).T) -
                np.dot(1 - self.output, np.log(1 - predicted_value).T)
        )
        return loss_value[0]

    def backward(self):
        d_loss_yh = - (
                np.divide(self.output, self.predicted_output) -
                np.divide(1 - self.output, 1 - self.predicted_output)
        )

        d_loss_z2 = d_loss_yh * d_sigmoid(self.cache['Z2'])
        d_loss_a1 = np.dot(self.param["W2"].T, d_loss_z2)
        d_loss_w2 = 1. / self.cache['A1'].shape[1] * np.dot(d_loss_z2, self.cache['A1'].T)
        d_loss_b2 = 1. / self.cache['A1'].shape[1] * np.dot(d_loss_z2, np.ones([d_loss_z2.shape[1], 1]))

        d_loss_z1 = d_loss_a1 * d_re_lu(self.cache['Z1'])
        # noinspection PyUnusedLocal
        d_loss_a0 = np.dot(self.param["W1"].T, d_loss_z1)
        d_loss_w1 = 1. / self.input.shape[1] * np.dot(d_loss_z1, self.input.T)
        d_loss_b1 = 1. / self.input.shape[1] * np.dot(d_loss_z1, np.ones([d_loss_z1.shape[1], 1]))

        self.param["W1"] = self.param["W1"] - self.lr * d_loss_w1
        self.param["b1"] = self.param["b1"] - self.lr * d_loss_b1
        self.param["W2"] = self.param["W2"] - self.lr * d_loss_w2
        self.param["b2"] = self.param["b2"] - self.lr * d_loss_b2

    def gd(self, epochs=3000):
        np.random.seed(1)

        self.init()

        for i in range(0, epochs):
            predicted_value, loss_value = self.forward
            self.backward()

            if i % 500 == 0:
                self.loss.append(loss_value)

    def predict(self, _input, _output):
        comp = np.zeros((1, _input.shape[1]))
        predicted_value, loss_value = self.pre(_input, _output)

        for i in range(0, predicted_value.shape[1]):
            if predicted_value[0, i] > self.threshold:
                comp[0, i] = 1
            else:
                comp[0, i] = 0

        acc = np.sum((comp == _output) / _input.shape[1])

        return comp, acc

    def pre(self, _input, _output):
        self.input = _input
        self.output = _output
        return self.forward

    def print_loss_graph(self, title, path):
        plt.clf()
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(self.loss)), self.loss)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.savefig(path)

    def print_plot_cf(self, _input, _output, title, path):
        plt.clf()
        self.input, self.output = _input, _output
        predicted, accuracy = self.predict(self.input, self.output)
        _input = np.around(np.squeeze(self.output), decimals=0).astype(np.int)
        _output = np.around(np.squeeze(predicted), decimals=0).astype(np.int)
        cf = confusion_matrix(_input, _output)
        plt.imshow(cf, cmap=plt.cm.Blues, interpolation='nearest')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.text(.25, 2.35, 'accuracy = %.2f' % accuracy, fontsize=14)
        tick_marks = np.arange(len(set(expected)))  # length of classes
        class_labels = ['0', '1']
        plt.xticks(tick_marks, class_labels)
        plt.yticks(tick_marks, class_labels)
        # plotting text value inside cells
        thresh = cf.max() / 2.
        for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
            plt.text(j, i, format(cf[i, j], 'd'), horizontalalignment='center',
                     color='white' if cf[i, j] > thresh else 'black')
        plt.savefig(path)


def parse_data(path, validation_divisor):
    data_frame = pd.read_csv(path, header=None, skiprows=1, dtype=int)
    columns = data_frame.columns[1:6]
    data_scaler = MinMaxScaler()
    scaled_frame = data_scaler.fit_transform(data_frame.iloc[:, 1:6])
    scaled_frame = pd.DataFrame(scaled_frame, columns=columns)
    _train_input = scaled_frame.iloc[0:validation_divisor, 0:4].values.transpose()
    _train_output = data_frame.iloc[0:validation_divisor, 5:].values.transpose()
    _validation_input = scaled_frame.iloc[validation_divisor:, 0:4].values.transpose()
    _validation_output = data_frame.iloc[validation_divisor:, 5:].values.transpose()
    return _train_input, _train_output, _validation_input, _validation_output


def predict_or():
    train_input, train_output, validation_input, validation_output = parse_data('or.csv', 9)
    or_network = DoubleLayerNetwork(train_input, train_output)
    or_network.lr = 0.005
    or_network.neurons = [4, 1, 1]
    or_network.gd(epochs=10000)
    or_network.print_loss_graph("loss rate", "or_loss.png")
    or_network.print_plot_cf(train_input, train_output, 'Or Training Set', "or_training.png")
    or_network.print_plot_cf(validation_input, validation_output, 'Or Validation Set', "or_validation.png")


def predict_xor():
    train_input, train_output, validation_input, validation_output = parse_data('xor.csv', 14)
    or_network = DoubleLayerNetwork(train_input, train_output)
    or_network.lr = 0.5
    or_network.neurons = [4, 9, 1]
    or_network.gd(epochs=10000)
    or_network.print_loss_graph("loss rate", "xor_loss.png")
    or_network.print_plot_cf(train_input, train_output, 'xOr Training Set', "xor_training.png")
    or_network.print_plot_cf(validation_input, validation_output, 'xOr Validation Set', "xor_validation.png")


predict_or()
predict_xor()

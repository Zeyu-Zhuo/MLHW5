# -*- coding: utf-8 -*-

import numpy as np
from numpynet.layer import Dense, ELU, ReLU, SoftmaxCrossEntropy
from numpynet.function import Softmax
from numpynet.utils import Dataloader, one_hot_encoding, load_MNIST, save_csv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

IntType = np.int64
FloatType = np.float64


class Model(object):
    """Model Your Deep Neural Network
    """
    def __init__(self, input_dim, output_dim):
        """__init__ Constructor

        Arguments:
            input_dim {IntType or int} -- Number of input dimensions
            output_dim {IntType or int} -- Number of classes
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_fn = SoftmaxCrossEntropy(axis=-1)
        self.build_model()
        self.label = None

    def build_model(self):
        """build_model Build the model using numpynet API
        """
        # TODO: Finish this function
        self.dense1 = Dense(self.input_dim, 256)
        self.dense2 = Dense(256, 128)
        self.dense3 = Dense(128, 64)
        self.dense4 = Dense(64, 32)
        self.dense5 = Dense(32,self.output_dim)
        self.elu1 = ELU(0.9)
        self.elu2 = ELU(0.9)
        self.elu3 = ELU(0.9)
        self.elu4 = ELU(0.9)
        
        
    def __call__(self, X):
        """__call__ Forward propogation of the model

        Arguments:
            X {np.ndarray} -- Input batch

        Returns:
            np.ndarray -- The output of the model. 
                You can return the logits or probits, 
                which depends on the way how you structure 
                the code.
        """
        # TODO: Finish this function
        out = self.dense1(X)
        out = self.elu1(out)
        out = self.dense2(out)
        out = self.elu2(out)
        out = self.dense3(out)
        out = self.elu3(out)
        out = self.dense4(out)
        out = self.elu4(out)
        logits = self.dense5(out)
        return logits
        raise NotImplementedError

    def bprop(self, logits, labels, istraining=True):
        """bprop Backward propogation of the model

        Arguments:
            logits {np.ndarray} -- The logits of the model output, 
                which means the pre-softmax output, since you need 
                to pass the logits into SoftmaxCrossEntropy.
            labels {np,ndarray} -- True one-hot lables of the input batch.

        Keyword Arguments:
            istraining {bool} -- If False, only compute the loss. If True, 
                compute the loss first and propagate the gradients through 
                each layer. (default: {True})

        Returns:
            FloatType or float -- The loss of the iteration
        """

        # TODO: Finish this function
        if(istraining):
            loss = self.loss_fn(logits,labels)
            grad = self.loss_fn.bprop()
            grad = self.dense5.bprop(grad)
            grad = self.dense4.bprop(self.elu4.bprop()*grad)
            grad = self.dense3.bprop(self.elu3.bprop()*grad)
            grad = self.dense2.bprop(self.elu2.bprop()*grad)
            grad = self.dense1.bprop(self.elu1.bprop()*grad)
            return loss
        # raise NotImplementedError

    def update_parameters(self, lr):
        """update_parameters Update the parameters for each layer.

        Arguments:
            lr {FloatType or float} -- The learning rate
        """
        # TODO: Finish this function
        self.dense1.update(lr)
        self.dense2.update(lr)
        self.dense3.update(lr)
        self.dense4.update(lr)
        self.dense5.update(lr)
        # raise NotImplementedError


def train(model,
          train_X,
          train_y,
          val_X,
          val_y,
          max_epochs=20,
          lr=1e-3,
          batch_size=16,
          metric_fn=accuracy_score,
          **kwargs):
    """train Train the model

    Arguments:
        model {Model} -- The Model object
        train_X {np.ndarray} -- Training dataset
        train_y {np.ndarray} -- Training labels
        val_X {np.ndarray} -- Validation dataset
        val_y {np.ndarray} -- Validation labels

    Keyword Arguments:
        max_epochs {IntType or int} -- Maximum training expochs (default: {20})
        lr {FloatType or float} -- Learning rate (default: {1e-3})
        batch_size {IntType or int} -- Size of each mini batch (default: {16})
        metric_fn {function} -- Metric function to measure the performance of 
            the model (default: {accuracy_score})
    """
    # TODO: Finish this function
    flag = True
    train_loss_res = []
    train_acc_res = []
    val_loss_res = []
    val_acc_res = []
    N_train = len(train_X)
    N_val = len(val_X)
    one_hot_train_y = one_hot_encoding(train_y)
    one_hot_val_y = one_hot_encoding(val_y)
    for index in range(max_epochs):
        train_loss, train_acc = 0, 0
        val_loss, val_acc = 0, 0
        idxs = np.arange(N_train)
        np.random.shuffle(idxs)
        temp_x, temp_y = train_X[idxs], one_hot_train_y[idxs]
        # training
        for i in range(0, N_train, batch_size):
            range_ = range(i, min(i+batch_size, N_train))
            logits = model(temp_x[range_])
            loss = model.bprop(logits, temp_y[range_])
            rightnum = model.loss_fn.rightnum
            train_loss += loss
            train_acc += rightnum
            model.update_parameters(lr)
        # validation

        train_loss /= N_train
        train_acc /= N_train
        train_loss_res.append(train_loss)
        train_acc_res.append(train_acc)

        for i in range(0, N_val, batch_size):
            range_ = range(i, min(i+batch_size, N_val))
            logits = model(val_X[range_])
            loss = model.bprop(logits,one_hot_val_y[range_])
            rightnum = model.loss_fn.rightnum
            val_loss += loss
            val_acc += rightnum
        val_loss /= N_val
        val_acc /= N_val
        val_loss_res.append(val_loss)
        val_acc_res.append(val_acc)
        if(val_acc>0.95 and flag):
            flag = False
            lr /= 10
        print("epoch: {}, train acc: {:.2f}%, train loss: {:.3f}, val acc: {:.2f}%, val loss: {:.3f}" .format(
            index+1, train_acc*100, train_loss, val_acc*100, val_loss))
    return train_loss_res, train_acc_res, val_loss_res, val_acc_res



def inference(model, X, y , batch_size=16, metric_fn=accuracy_score, **kwargs):
    """inference Run the inference on the given dataset

    Arguments:
        model {Model} -- The Neural Network model
        X {np.ndarray} -- The dataset input
        y {np.ndarray} -- The sdataset labels

    Keyword Arguments:
        metric {function} -- Metric function to measure the performance of the model 
            (default: {accuracy_score})

    Returns:
        tuple of (float, float): A tuple of the loss and accuracy
    """

    # TODO: Finish this function
    logits = model(X)
    labels = one_hot_encoding(y)
    loss = model.bprop(logits, labels)
    rightnum = model.loss_fn.rightnum
    accuracy = rightnum / X.shape[0]
    return accuracy, loss,logits.argmax(axis=1)
    raise NotImplementedError

def inferenceNoy(model, X, batch_size=16, metric_fn=accuracy_score, **kwargs):
    """inference Run the inference on the given dataset

    Arguments:
        model {Model} -- The Neural Network model
        X {np.ndarray} -- The dataset input
        y {np.ndarray} -- The sdataset labels

    Keyword Arguments:
        metric {function} -- Metric function to measure the performance of the model 
            (default: {accuracy_score})

    Returns:
        tuple of (float, float): A tuple of the loss and accuracy
    """

    # TODO: Finish this function
    logits = model(X)
    return logits.argmax(axis=1)
    raise NotImplementedError



def main():
    print('loading data #####')
    train_X, train_y = load_MNIST(path ='dataset/',name="train")
    val_X, val_y = load_MNIST(path = 'dataset/', name="val")
    test_X = load_MNIST(path = 'dataset/', name="test")
    test_loss, test_acc = None, None
    print('loading data complete #####')
    # TODO: 1. Build your model
    # TODO: 2. Train your model with training dataset and
    #       validate it  on the validation dataset
    # TODO: 3. Test your trained model on the test dataset
    #       you need have at least 95% accuracy on the test dataset to receive full scores

    # Your code starts here

    #NOTE: WE HAVE PROVIDED A SKELETON FOR THE MAIN FUNCTION. FEEL FREE TO CHANGE IT AS YOU WISH, THIS IS JUST A SUGGESTED FORMAT TO HELP YOU.

    batchSize = 32
    learningRate = 0.1
    model = Model(input_dim = 784,output_dim = 10)

    print('Model built #####')
    train_loss, train_acc,val_loss, val_acc  = train(model, train_X, train_y, val_X, val_y, max_epochs=200, lr=learningRate, batch_size=batchSize, metric_fn=accuracy_score)
    print('Training complete #####')

    # Plot of train and val accuracy vs iteration
    plt.figure(figsize=(10,7))
    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations')
    plt.title('Accuracy vs number of iterations')
    plt.plot(np.linspace(0,199,200), train_acc, label = 'Train accuracy across iterations')
    plt.plot(np.linspace(0,199,200), val_acc, label = 'Val accuracy across iterations')
    plt.legend(loc = 'upper right')
    plt.show()
    # Plot of train and val loss vs iteration
    plt.figure(figsize=(10,7))
    plt.ylabel('Loss')
    plt.xlabel('Number of iterations')
    plt.title('Loss vs number of iterations')
    plt.plot(np.linspace(0,199,200), train_loss, label = 'Train loss across iterations')
    plt.plot(np.linspace(0,199,200), val_loss, label = 'Val loss across iteration')
    plt.legend(loc='upper right')
    plt.show()

    # Implement inference such that you predict the labels and also evaluate val_accuracy and loss if true labels are provided
    val_acc, val_loss, val_pred = inference(model, val_X, val_y, batch_size = batchSize)

    # Inference on test dataset without labels

    #Implement inference function so that you can return the test prediction output and save it in test_pred. You are allowed to create a different function to generate just the predicted labels.
    test_pred = inferenceNoy(model, test_X, batch_size = batchSize)
    
    save_csv(test_pred)
    # Your code ends here

    print("Validation loss: {0}, Validation Acc: {1}%".format(val_loss, 100 * val_acc))
    if val_acc > 0.95:
        print("Your model is well-trained.")
    else:
        print("You still need to tune your model")


if __name__ == '__main__':
    main()

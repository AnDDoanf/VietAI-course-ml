from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

import importlib.util
import sys
import numpy as np

import signal
from contextlib import contextmanager
import tensorflow as tf

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

spec = importlib.util.spec_from_file_location("module.name", sys.argv[1])
submitted_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submitted_module)

# Ground-truth functions
# GRADED FUNCTION: normalize
def normalize(train_x, val_x, test_x):
    """TODO 1: normalize
    This function computes the mean and standard deviation of all pixels and performs data scaling on train_x, val_x and test_x using these computed values.
    Note that in this classification problem, the shape of the data is (num_samples, image_width * image_height).

    :param train_x: train images, shape=(num_train, image_height * image_width)
    :param val_x: validation images, shape=(num_val, image_height * image_width)
    :param test_x: test images, shape=(num_test, image_height * image_width)
    """
    # The shape of train_mean and train_std should be (1, 1)
    ### START CODE HERE ### (≈5 lines)
    train_mean = np.mean(train_x, axis=(0,1), dtype=np.float64, keepdims=True)
    train_std = np.std(train_x, axis=(0,1), dtype=np.float64, keepdims=True)

    train_x = (train_x-train_mean)/train_std
    val_x = (val_x-train_mean)/train_std
    test_x = (test_x-train_mean)/train_std
    ### END CODE HERE ###
    return train_x, val_x, test_x


# GRADED FUNCTION
class SoftmaxClassifier(object):
    def __init__(self, w_shape):
        """__init__

        :param w_shape: create w with shape w_shape using normal distribution
        """
        self.w = np.random.normal(0, np.sqrt(2. / np.sum(w_shape)), w_shape)

    def softmax(self, x):
        """TODO 2: softmax

        :param x: input
        """
        result = None
        ### START CODE HERE ### (≈4 lines)
        max_x = np.max(x, axis=1, keepdims=True)
        x -= max_x
        x_exp = np.exp(x)
        result = x_exp / np.sum(x_exp, axis=1, keepdims=True)
        ### END CODE HERE ###
        return result

    def feed_forward(self, x):
        """TODO 3: feed_forward
        This function computes the output of your softmax regression model

        :param x: input
        """
        result = None
        ### START CODE HERE ### (≈2 lines)
        x_out = np.dot(x, self.w)
        result = self.softmax(x_out)
        ### END CODE HERE ###
        return result

    def compute_loss(self, y, y_hat):
        """TODO 4: compute_loss
        Compute the loss using y (label) and y_hat (predicted class)

        :param y:  the label, the actual class of the sample data
        :param y_hat: the classifying probabilities of all sample data
        """
        loss = 0
        ### START CODE HERE ### (≈3 lines)
        loss = -np.log(y_hat)
        loss = np.sum(loss * y, axis=1)
        loss = np.mean(loss)
        ### END CODE HERE ###
        return loss

    def get_grad(self, x, y, y_hat):
        """TODO 5: get_grad
        Compute and return the gradient of w

        :param loss: computed loss between y_hat and y in the train dataset
        :param y_hat: predicted y
        """
        w_grad = None
        ### START CODE HERE ### (≈2 lines)
        num_x = x.shape[0]
        w_grad = np.dot(x.T, -(y - y_hat)) / num_x
        ### END CODE HERE ###
        return w_grad

    def update_weight(self, grad, learning_rate):
        """update_weight
        Update w using the computed gradient.

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        """
        self.w = self.w - learning_rate * grad
        return self.w

    def update_weight_momentum(self, grad, learning_rate, momentum, momentum_rate):
        """update_weight using momentum
        BONUS:[YC1.8]
        Update w using the algorithm with momentum

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        :param momentum: the array storing momentum for training w, should have the same shape as that of w
        :param momentum_rate: float, how much momentum to reuse after each loop (denoted as gamma in the following section)
        """
        momentum *= momentum_rate
        momentum += learning_rate * grad
        self.w = self.w - momentum
        return self.w

    def numerical_check(self, x, y, grad):
        i = 3
        j = 0
        eps = 0.000005
        w_test0 = np.copy(self.w)
        w_test1 = np.copy(self.w)
        w_test0[i, j] = w_test0[i, j] - eps
        w_test1[i, j] = w_test1[i, j] + eps

        y_hat0 = np.dot(x, w_test0)
        y_hat0 = self.softmax(y_hat0)
        loss0 = self.compute_loss(y, y_hat0)

        y_hat1 = np.dot(x, w_test1)
        y_hat1 = self.softmax(y_hat1)
        loss1 = self.compute_loss(y, y_hat1)

        numerical_grad = (loss1 - loss0) / (2 * eps)
        # print(numerical_grad)
        # print(grad[i, j])


# GRADED FUNCTION
def is_stop_training(all_val_loss, patience=5):
    """TODO 6:  is_stop_training
    Check whether training need to be stopped

    :param all_val_loss: list of all validation loss values during training
    """
    is_stopped = False
    ### START CODE HERE ###
    num_val_increase = 0
    if (len(all_val_loss) < 2):
        return False
    for i in range(len(all_val_loss) - 1,0, -1):
        if all_val_loss[i] > all_val_loss[i-1]:
            num_val_increase += 1
        if num_val_increase >= patience:
            return True
    ### END CODE HERE ###
    return is_stopped

# GRADED FUNCTION
def softmax_test(y_hat, test_y):
    """TODO 7: test
    Compute the confusion matrix based on labels and predicted values

    :param classifier: the trained classifier
    :param y_hat: predicted probabilites, output of classifier.feed_forward
    :param test_y: test labels
    """

    y_hat = np.argmax(y_hat, axis=1)
    test_y = np.argmax(test_y, axis=1)
    confusion_mat = np.zeros((10,10))
    ### START CODE HERE ###
    for i in range(10):
        class_i_idx = test_y == i
        num_class_i = np.sum(class_i_idx)
        y_hat_i = y_hat[class_i_idx]
        for j in range(10):
            confusion_mat[i,j] = 1.0*np.sum(y_hat_i == j)/num_class_i
    ### END CODE HERE ###
    return confusion_mat

# GRADED FUNCTION
class SoftmaxRegressionTF(tf.keras.Model):
    def __init__(self, num_class):
        super(SoftmaxRegressionTF, self).__init__()
        # TODO 8: init all weights
        ### START CODE HERE ###
        self.dense = tf.keras.layers.Dense(num_class, kernel_initializer=tf.keras.initializers.RandomNormal(seed=2020))
        ### END CODE HERE ###

    def call(self, inputs, training=None, mask=None):
        # TODO 9: implement your feedforward
        ### START CODE HERE ###
        output = self.dense(inputs)
        ### END CODE HERE ###
        try:
          output = tf.nn.softmax(output)
        except: # if softmax op does not exist on the gpu
          with tf.device('/cpu:0'):
              output = tf.nn.softmax(output)

        return output


if __name__ == "__main__":
    score = 0
    msg = ""

    ex_name = "TODO 1"
    score_todo = 0
    for i in range(10):
        train_x = np.random.rand(3,1024)
        e_train, e_val, e_test = normalize(train_x,train_x,train_x)
        try:
            with time_limit(10):
                o_train, o_val, o_test = submitted_module.normalize(train_x,train_x,train_x)
                score_todo += int((np.sum(np.abs(e_train-o_train)+np.sum(np.abs(e_val-o_val))+np.sum(np.abs(e_test-o_test))))==0)
        except TimeoutException as e:
            msg += ex_name + ": " + str(e) + '\n'
            break
        except Exception as e:
            msg += ex_name + ": " + str(e) + '\n'
            break

    msg += ex_name + ": " + str(score_todo) + "\n"
    score += score_todo

    score_2 = 0
    score_3 = 0
    score_4 = 0
    score_5 = 0

    try:
        w_shape = (15, 10)
        w = np.random.normal(0, np.sqrt(2. / np.sum(w_shape)), w_shape)
        e_Softmax = SoftmaxClassifier(w_shape)
        e_Softmax.w = w
        o_Softmax = submitted_module.SoftmaxClassifier(w_shape)
        o_Softmax.w = w

        for i in range(10):
            x = np.random.rand(5, 15)
            y = np.random.randint(2, size=50).reshape(5, 10)
            e_soft = e_Softmax.softmax(x)
            e_out = e_Softmax.feed_forward(x)
            e_loss = e_Softmax.compute_loss(y,e_out)
            e_grad = e_Softmax.get_grad(x,y,e_out)

            try:
                with time_limit(2):
                   o_soft = o_Softmax.softmax(x)
                   score_2 += int(np.sum(np.abs(e_soft-o_soft)) == 0)
            except TimeoutException as e:
                msg += "TODO 2: " + str(e) + '\n'
                break
            except Exception as e:
                msg += "TODO 2: " + str(e) + '\n'
                break

            try:
                with time_limit(2):
                    o_out = o_Softmax.feed_forward(x)
                    score_3 += int(np.sum(np.abs(e_out-o_out))==0)
            except TimeoutException as e:
                msg += "TODO 3: " + str(e) + '\n'
                break
            except Exception as e:
                msg += "TODO 3: " + str(e) + '\n'
                break

            try:
                with time_limit(2):
                    o_loss = o_Softmax.compute_loss(y,e_out)
                    score_4 += int((e_loss - o_loss) == 0)
            except TimeoutException as e:
                msg += "TODO 4: " + str(e) + '\n'
                break
            except Exception as e:
                msg += "TODO 4: " + str(e) + '\n'
                break

            try:
                with time_limit(2):
                    o_grad = o_Softmax.get_grad(x,y,e_out)
                    score_5 += int(np.sum(np.abs(e_grad - o_grad)) == 0)
            except TimeoutException as e:
                msg += "TODO 5: " + str(e) + '\n'
                break
            except Exception as e:
                msg += "TODO 5: " + str(e) + '\n'
                break

    except Exception as e:
        msg += "TODO 2-5: " + str(e) + '\n'

    msg += "TODO 2: " + str(score_2) + "\n"
    msg += "TODO 3: " + str(score_3) + "\n"
    msg += "TODO 4: " + str(score_4) + "\n"
    msg += "TODO 5: " + str(score_4) + "\n"

    score += score_5 + score_4 + score_3 + score_2

    ex_name = "TODO 6"
    score_todo = 0

    for i in range(10):
        all_val_loss = list(np.random.uniform(low=1,high=50,size=10))
        expected = is_stop_training(all_val_loss,patience=5)
        try:
            ouput = submitted_module.is_stop_training(all_val_loss,patience=5)
            score_todo += (expected==ouput)
        except TimeoutException as e:
            msg += ex_name+ ": " + str(e) + '\n'
            break
        except Exception as e:
            msg += ex_name + ": " + str(e) + '\n'
            break

    msg += ex_name + ": " + str(score_todo) + "\n"
    score += score_todo

    ex_name = "TODO 7"
    score_todo = 0

    for i in range(10):
        y_hat = tf.nn.softmax(np.random.rand(20,10)).numpy()
        test_y = np.eye(20,10)

        expected = softmax_test(y_hat,test_y)
        try:
            ouput = submitted_module.softmax_test(y_hat,test_y)
            score_todo += int((np.sum(np.abs(expected-ouput))==0)*2)
        except TimeoutException as e:
            msg += ex_name + ": " + str(e) + '\n'
            break
        except Exception as e:
            msg += ex_name + ": " + str(e) + '\n'
            break

    msg += ex_name + ": " + str(score_todo) + "\n"
    score += score_todo

    score_8 = 0

    try:
        e_SoftmaxTF = SoftmaxRegressionTF(10)
        o_SoftmaxTF = submitted_module.SoftmaxRegressionTF(10)

        for i in range(10):
            x = tf.random.uniform((5,15),minval=-5, maxval=5,seed=2020)
            e_out = e_SoftmaxTF(x).numpy()

            try:
                o_out = o_SoftmaxTF(x).numpy()
                score_8 += int((np.sum(np.abs(e_out - o_out)) == 0)*2)
            except TimeoutException as e:
                msg +=  "TODO 8: " + str(e) + '\n'
                break
            except Exception as e:
                msg += "TODO 8: " + str(e) + '\n'
                break

    except Exception as e:
        msg += "TODO 8: " + str(e) + '\n'

    msg += "TODO 8: " + str(int(score_8)) + "\n"
    score += int(score_8)

    print(score)
    print(msg)









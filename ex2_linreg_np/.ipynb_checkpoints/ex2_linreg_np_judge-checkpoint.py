import importlib.util
import sys

import signal
from contextlib import contextmanager
import random
import numpy as np
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

def np_mse_loss(w, b, X, y):
    '''
    Evaluate the loss function in a non-vectorized manner for 
    inputs `X` and targets `y`, at weights `w` and `b`.
    '''
    
    losss = 0
    N = len(y)
    for i in range(len(y)):
        
        # TO_DO_1: complete below expression to calculate loss function
        y_hat = w[0] * X[i, 0] + w[1] * X[i, 1] + b
        losss += (y_hat - y[i])**2
        
    return losss / (2 * N)

def np_loss_vectorized(w, b, X, y):
    '''
    Evaluate the loss function in a vectorized manner for 
    inputs `X` and targets `t`, at weights `w1`, `w2` and `b`.
    '''
    
    #TO_DO_2: Complete the following expression
    N = len(y)
    w = np.asarray(w)
    y_hat = np.dot(X, w) + b
    
    return np.sum((y_hat - y)**2) / (2.0 * N)

def np_solve_exactly(X, y):
    '''
    Solve linear regression exactly. (fully vectorized)
    
    Given `X` - NxD matrix of inputs
          `t` - target outputs
    Returns the optimal weights as a D-dimensional vector

    '''
    
    #TO_DO_3: Complete the below followed the above expressions
    A = np.dot(X.T, X)
    c = np.dot(X.T, y)
    
    return np.dot(np.linalg.inv(A), c)


def np_grad_fn(weights, X, y):
    '''
    Given `weights` - a current "Guess" of what our weights should be
          `X` - matrix of shape (N,D) of input features
          `t` - target y values
    Return gradient of each weight evaluated at the current value
    '''
    #TO_DO_4: Complete the below followed the above expressions
    N, D = np.shape(X)
    y_hat = np.dot(X, weights)
    error = y_hat - y
    return np.dot(np.transpose(X), error) / N

def np_solve_via_gradient_descent(X, y, niter=100000, alpha=0.005):
    '''
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
    Solves for linear regression weights.
    Return weights after `niter` iterations.
    '''
    N, D = np.shape(X)
    # initialize all the weights to zeros
    w = np.zeros([D])
    for k in range(niter):
        #TO_DO_5: Complete the below followed the above expressions
        dw = np_grad_fn(w, X, y)
        w = w - alpha*dw
    return w
score = 0
msg = ""

##### BAI 1 ####
#ex_name = "TO_DO_1"
#theta = np.random.rand(3, 1)
#model = linreg(theta)
#for i in range(10):
#    x = np.random.rand(3, 1)
#    expected_output = model.forward(x)

#    try:
#        with time_limit(1):
#            submitted_class = submitted_module.linreg(theta)
#            submitted_output = submitted_class.forward(x)
#            score += int( np.sum(np.abs(submitted_output - expected_output))==0)
#    except TimeoutException as e:
#        msg += ex_name + ": "+ str(e) + '\n'
#        break
#    except Exception as e:
#        msg += ex_name + ": "+ str(e) + '\n'
#        break

ex_name = "TO_DO_1"
score_todo = 0
for i in range(10):
    w = np.random.rand(2)
    b = 10*np.random.rand()
    X = np.random.rand(100, 2)
    y = np.random.rand(100)
    
    expected_output = np_mse_loss(w, b, X, y)

    try:
        with time_limit(1):
            submitted_output = submitted_module.np_mse_loss(w, b, X, y)
            score_todo += int( np.sum(np.abs(submitted_output - expected_output))<0.0001)
    except TimeoutException as e:
        msg += ex_name + ": "+ str(e) + '\n'
        break
    except Exception as e:
        msg += ex_name + ": "+ str(e) + '\n'
        break
score += score_todo
msg += ex_name + ": " + str(score_todo) + "\n"


ex_name = "TO_DO_2"
score_todo = 0
for i in range(10):
    w = np.random.rand(2)
    b = 10*np.random.rand()
    X = np.random.rand(10000, 2)
    y = np.random.rand(10000)
    
    expected_output = np_loss_vectorized(w, b, X, y)

    try:
        with time_limit(1):
            submitted_output = submitted_module.np_loss_vectorized(w, b, X, y)
            score_todo += int( np.sum(np.abs(submitted_output - expected_output))<0.0001)
    except TimeoutException as e:
        msg += ex_name + ": "+ str(e) + '\n'
        break
    except Exception as e:
        msg += ex_name + ": "+ str(e) + '\n'
        break
score += score_todo
msg += ex_name + ": " + str(score_todo) + "\n"


ex_name = "TO_DO_3"
score_todo = 0
for i in range(10):
    X = np.random.rand(100, 2)
    y = np.random.rand(100)
    
    expected_output = np_solve_exactly(X, y)

    try:
        with time_limit(1):
            submitted_output = submitted_module.np_solve_exactly(X, y)
            score_todo += int( np.sum(np.abs(submitted_output - expected_output))<0.0001)
    except TimeoutException as e:
        msg += ex_name + ": "+ str(e) + '\n'
        break
    except Exception as e:
        msg += ex_name + ": "+ str(e) + '\n'
        break
score += score_todo
msg += ex_name + ": " + str(score_todo) + "\n"


ex_name = "TO_DO_4"
score_todo = 0
for i in range(10):
    weights = np.zeros(2)
    X = np.random.rand(10000, 2)
    y = np.random.rand(10000)
    
    expected_output = np_grad_fn(weights, X, y)

    try:
        with time_limit(1):
            submitted_output = submitted_module.np_grad_fn(weights, X, y)
            score_todo += int( np.sum(np.abs(submitted_output - expected_output))<0.0001)
    except TimeoutException as e:
        msg += ex_name + ": "+ str(e) + '\n'
        break
    except Exception as e:
        msg += ex_name + ": "+ str(e) + '\n'
        break
score += score_todo
msg += ex_name + ": " + str(score_todo) + "\n"


ex_name = "TO_DO_5"
score_todo = 0
for i in range(10):
    X = np.random.rand(10000, 2)
    y = np.random.rand(10000)
    
    expected_output = np_solve_via_gradient_descent(X, y, 10, 0.005)

    try:
        with time_limit(1):
            submitted_output = submitted_module.np_solve_via_gradient_descent(X, y, 10, 0.005)
            score_todo += int( np.sum(np.abs(submitted_output - expected_output))<0.0001)
    except TimeoutException as e:
        msg += ex_name + ": "+ str(e) + '\n'
        break
    except Exception as e:
        msg += ex_name + ": "+ str(e) + '\n'
        break
score += score_todo
msg += ex_name + ": " + str(score_todo) + "\n"

print(score)
print(msg)

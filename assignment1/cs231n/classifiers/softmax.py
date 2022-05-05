from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N=X.shape[0]
    C=W.shape[1]
    for i in range(N):
      s=X[i].dot(W)
      s=np.exp(s)
      loss+=-np.log(s[y[i]]/np.sum(s))
      for j in range(C):
        if(j==y[i]):
          dW[:,j]+=X[i].transpose()*-(np.sum(s)-s[y[i]])/np.sum(s)
        else:
          dW[:,j]+=X[i].transpose()*s[j]/np.sum(s)
    dW/=N
    dW+=W*2*reg
    loss/=N
    loss+=reg*np.sum(W*W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N=X.shape[0]
    D=X.shape[1]
    C=W.shape[1]
    S=X.dot(W)    
    S_exp=np.exp(S)
    S_exp_sum=np.sum(S_exp,axis=1)
    x_index=np.arange(0,N)
    syi=S_exp[x_index,y]
    s=syi/S_exp_sum
    s_loss=-np.log(s)
    loss=np.sum(s_loss)/N
    loss+=reg*np.sum(W*W)
    grad_L=1
    grad_L/=N
    grad_s_loss=np.ones((N,1))*grad_L
    grad_s=grad_s_loss*(-1/s.reshape((-1,1)))
    # print(grad_s)
    grad_S_exp=-syi.reshape((-1,1)).dot(np.ones((1,C)))
    temp=S_exp_sum-syi
    grad_S_exp[x_index,y]=temp
    temp=S_exp_sum*S_exp_sum
    grad_S_exp=grad_S_exp/temp.reshape((-1,1))
    grad_S_exp*=grad_s
    grad_S=grad_S_exp*S_exp
    grad_W1=X.transpose().dot(grad_S)
    dW=grad_W1+2*reg*W
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

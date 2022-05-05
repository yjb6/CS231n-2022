from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    N = X.shape[0]
    C = W.shape[1]
    D = W.shape[0]
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    m=np.zeros((N,C))
    grad_w3=np.zeros((D,C))
    dic={}
    for i in range(num_train):
        grad_temp=np.zeros((D,C))
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            
            if margin > 0:
              dW[:,j]+=X[i].transpose()
              dW[:,y[i]]+=-X[i].transpose()
              grad_w3[:,j]+=X[i].transpose()
              grad_w3[:,y[i]]+=-X[i].transpose()
              m[i, j] = 1
              loss += margin
        dic[i]=grad_temp/num_train

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW/=num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW+=reg*W*2
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=X.shape[0]
    D=X.shape[1]
    C=W.shape[1]
    Z=X.dot(W)
    loss=0.0
    x_index=np.arange(0,N)
    target_score=Z[x_index,y]
    Z-=target_score.reshape((-1,1))
    Z+=1
    Z[x_index,y]=0
    loss+=np.sum(Z*(Z>0))
    loss /= N
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    grad_L=1.0
    grad_l1=grad_L/N
    grad_z=np.ones((N,C))*grad_l1
    grad_z=grad_z*(Z>0)
    grad_z[x_index,y]=-np.sum(grad_z,axis=1)
    X_T=X.transpose()
    grad_w=X_T.dot(grad_z)
    grad_l2=reg*grad_L
    grad_w+=W*2*grad_l2
    dW=grad_w
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

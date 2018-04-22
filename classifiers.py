import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    regularization!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    '''
    N, C = logits.shape
    maxval = -1.0 * np.max(logits, axis = 1)
    maxval = np.tile(maxval, C).reshape(C, N).transpose()
    val = logits + maxval
    expval = np.exp(val)
    sum = np.sum(expval, axis = 1)
    pr = (expval.transpose() / sum).transpose()

    logterm = pr[[range(N)], y]
    loss = np.sum(-1.0 * np.log(logterm))/ N


    pr[[range(N)], y] -= 1
    dlogits = pr / N
    '''

    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits /= N

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits

'''
#print (np.max(np.array([[1,2],[3,4]]), axis = 0))
m = np.array([[1,2],[1,4],[2,6],[2,8]])
print (m[[range(4)],[0,1,0,1]])

print (list(map(lambda x: np.sum(np.where(m[:,1] == x)), [1,2])))


#print (m[[0,1,2,3],[1,1,1,1,1]] + 1)
'''

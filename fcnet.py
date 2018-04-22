import numpy as np

from classifiers import softmax
from layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)



def random_init(n_in, n_out, weight_scale=5e-2, dtype=np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: The number of input nodes into each output.
    - n_out: The number of output nodes for each input.
    """
    W = None
    b = None

    #np.random.seed(2)

    weight_scale = 2.0/ (n_in + n_out)
    #weight_scale = 6e-2

    #xavier_uniform = np.sqrt(6)/ (np.sqrt(n_in + n_out))

    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    b = np.zeros(n_out, dtype= dtype)
    W = np.random.normal(0, weight_scale, [n_in, n_out])
    #W = np.ones((n_in, n_out)) * 0.01
    #W = W.astype(dtype)
    #W = np.random.uniform(-xavier_uniform, xavier_uniform, [n_in, n_out])
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return W, b



class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:
    [linear - relu - (dropout)] x (N - 1) - linear - softmax
    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10,
                 dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                 seed=None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: A list of the size of each hidden layer
        - input_dim: A list giving the size of the input
        - num_classes: Number of classes to classify.
        - dropout: A scalar between 0. and 1. determining the dropout factor.
        If dropout = 0., then dropout is not applied.
        - reg: Regularisation factor.

        """
        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed:
            np.random.seed(seed)
        self.params = dict()
        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the Xavier
        initialisation (see manual).
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        full_dims = [input_dim] + hidden_dims + [num_classes]

        for dim_inx in range(1, len(full_dims)):
            inx_str = str(dim_inx)

            self.params['W' + inx_str], self.params['b' + inx_str] = \
                            random_init(full_dims[dim_inx - 1], full_dims[dim_inx])

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # When using dropout we need to pass a dropout_param dictionary to
        # each dropout layer so that the layer knows the dropout probability
        # and the mode (train / test). You can pass the same dropout_param to
        # each dropout layer.
        self.dropout_params = dict()
        if self.use_dropout:
            self.dropout_params = {"train": True, "p": dropout}
            if seed is not None:
                self.dropout_params["seed"] = seed
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Args:
        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()
        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        VAL = X.copy()

        for i in range(1, self.num_layers):
            linear_cache['L{}'.format(i)] = linear_forward(VAL, self.params['W{}'.format(i)], self.params['b{}'.format(i)])
            relu_cache['R{}'.format(i)] = relu_forward(linear_cache['L{}'.format(i)])
            if self.use_dropout:
                dropout_cache['D{}'.format(i)], dropout_cache['MASK{}'.format(i)] = dropout_forward(relu_cache['R{}'.format(i)],\
                                                                 self.dropout_params['p'], self.dropout_params['train'],\
                                                                 self.dropout_params['seed'])
                VAL = dropout_cache['D{}'.format(i)]
            else:
                VAL = relu_cache['R{}'.format(i)]


        linear_cache['L{}'.format(self.num_layers)] = linear_forward(VAL, self.params['W{}'.format(self.num_layers)],\
                                                           self.params['b{}'.format(self.num_layers)])


        scores = linear_cache['L{}'.format(self.num_layers)]



        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, grads = 0, dict()


        """
        TODO: Implement the backward pass for the fully-connected net. Store
        the loss in the loss variable and all gradients in the grads
        dictionary. Compute the loss with softmax. grads[k] has the gradients
        for self.params[k]. Add L2 regularisation to the loss function.
        NOTE: To ensure that your implementation matches ours and you pass the
        automated tests, make sure that your L2 regularization includes a
        factor of 0.5 to simplify the expression for the gradient.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################


        loss, grad = softmax(scores, y)

        if self.use_dropout:
            VAR = dropout_cache['D{}'.format(self.num_layers - 1)]
        else:
            VAR = relu_cache['R{}'.format(self.num_layers - 1)]

        dX, grads['W{}'.format(self.num_layers)], grads['b{}'.format(self.num_layers)] = linear_backward(grad, \
            VAR, self.params['W{}'.format(self.num_layers)],self.params['b{}'.format(self.num_layers)])

        grads['W{}'.format(self.num_layers)] += self.reg * self.params['W{}'.format(self.num_layers)]

        loss += 0.5 * self.reg * np.sum(self.params['W' + str(self.num_layers)] ** 2)


        for inx in range(self.num_layers-1, 0, -1):
            if self.use_dropout:
                dX = dropout_backward(dX, dropout_cache['MASK{}'.format(inx)], self.dropout_params['p'])

            dX = relu_backward(dX, linear_cache['L' + str(inx)])

            if inx - 1 != 0:
                if self.use_dropout:
                    pre_layer = dropout_cache['D{}'.format(inx - 1)]
                else:
                    pre_layer = relu_cache['R{}'.format(inx - 1)]
                dX, grads['W' + str(inx)], grads['b' + str(inx)] = linear_backward(dX, pre_layer,
                                                        self.params['W{}'.format(inx)], self.params['b{}'.format(inx)])

                grads['W' + str(inx)] += self.reg * self.params['W' + str(inx)]
                loss += 0.5 * self.reg * np.sum(self.params['W' + str(inx)] ** 2)


            else:

                dX, grads['W' + str(inx)], grads['b' + str(inx)] = linear_backward(dX, X, self.params['W{}'.format(inx)],
                                                                                   self.params['b{}'.format(inx)])
                grads['W' + str(inx)] += self.reg * self.params['W' + str(inx)]
                loss += 0.5 * self.reg * np.sum(self.params['W' + str(inx)] ** 2)

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        return loss, grads

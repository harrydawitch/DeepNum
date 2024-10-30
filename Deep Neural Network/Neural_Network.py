import numpy as np
import copy

class neural_network():
    '''
    Represents a neural network model with functionality for forward and backward propagation, 
    caching, parameter management, and various optimization, loss settings.

    This class serves as a foundational neural network implementation that can be customized 
    for different types of architectures and optimization strategies. It includes mechanisms 
    for storing parameters, caches for forward and backward passes, and supports features like 
    optimizers, loss functions, and metrics for model evaluation.

    
    Attributes: 
        forward_cache (dict): Stores the feedforward cache for each layer (e.g., Z, A), used for backpropagation and prediction.
        backward_cache (dict): Stores the backpropagation cache for each layer (e.g., dZ, dW, db), used when updating parameters.
        parameters (dict): Contains model parameters for each layer (e.g., W, b), which will be updated during training.
        best_parameters (dict or None): Holds a copy of parameters achieving the lowest loss; initialized as None.
        
        layers (int): Tracks the number of layers in model
        activation (list): Contain activation function for each layer as a list of string 
        regularizers (list): A list of tuples contain regularizer hyperparameters

        compiled_optimizer (str or None): Name of the optimizer used for training; initialized as None.
        compiled_loss (str or None): Name of the loss function used for training; initialized as None.
        compiled_metric (str or None): Name of the metric used for evaluation; initialized as None.

        velocity (dict): Contains velocity parameters for each layer when using momentum-based optimizers.
        EWMA (dict): Contains Exponentially Weighted Moving Average (EWMA) parameters for each layer (used in RMSprop).
        moments (dict): Contains first and second moments for each layer when using the Adam optimizer.
        t (int): Tracks the time step `t` for adaptive optimizers like Adam.


    Raises:
        Exception: If an invalid activation function name is provided.


    Usage:
        The `Neural_network` class is intended to be used for building and training neural network models. 
        Parameters and layers should be configured through additional methods, and an optimizer and loss function can be compiled to train the model effectively.
    '''

    def __init__(self):
        self.forward_cache = {}     
        self.backward_cache = {} 
        self.parameters= {}
        self.best_parameters= None

        self.layers = 0
        self.activation = [0]
        self.regularizers = [0]

        self.compiled_optimizer = None
        self.compiled_loss = None
        self.compiled_metric = None

        self.velocity = {}
        self.EWMA= {}
        self.moments= {}
        self.t = 0


    def Dense(self, units= None, input_dim= None, activation= 'linear', initializers= 'constant', regularizer= (None, 0)):
        '''
        Initializes the parameters (W, b) for a fully connected (dense) layer and specifies its activation function, weights initialization and regularization.

        This function add a dense layer to the neural network model by setting up its weight (W) and bias (b) parameters.
        The activation function for the layer is also specified here. The initialization and regularization are optional

        Arguments:

            units (int): Number of neurons in this layer
            input_dim (int): The input dimension for this layer (Only if this is the first layer)
            activation (str): The activation function for this layer (e.g., 'relu', 'sigmoid', 'softmax', 'tanh', or 'linear'.)
            initializers (str): The weights initialization for this layer (e.g., 'constant', he_normal, 'he_uniform', 'glorot_normal', 'glorot_uniform')
            regularizer (tuple/list): A list or a tuple values that contains (name of regularizer, lambda value) (e.g., 'l2', 0.5)
        '''

        self.layers += 1    # After calling this function the number of layers should increment by 1


        # If user doesn't specify input_dim, it means this is not the first layer, so we automatically use the previous layer's dimension
        if input_dim is not None: 
            n = input_dim
    
        else:                     
            if f'W{self.layers - 1}' not in self.parameters:            
                raise ValueError(f"Please specify input dimension for the first layer!")
            
            else:
                n = self.parameters[f"W{self.layers - 1}"].shape[0]


        # Initialize model parameters and store them in the parameters class attribute. 
        weight = np.random.randn(units, n) * self.initialization(initializers= initializers, n_in= n, n_out= units, shape= (units, n))
        bias = np.zeros((units, 1))

        self.parameters[f'W{self.layers}'] = weight
        self.parameters[f'b{self.layers}'] = bias


        # Append the activation function name to the activation class attribute
        if activation == 'relu' or activation == 'sigmoid' or activation == 'softmax' or activation == 'tanh' or activation == 'linear':
            self.activation.append(activation)
        else:
            raise Exception("Please provide the correct activation function!")

        # Checking regularizer input format and append the regularizers name and it lambda to the regularizers class attribute
        if isinstance(regularizer, (list, tuple)) and len(regularizer) == 2:
            self.regularizers.append(regularizer)
        else:
            raise Exception('Please provide the correct regularizer ("l1", "l2") and its correspond lambda')


    def tanh(self, X, derivative= False):
        '''
        Transforms input data into the hyperbolic tangent (tanh) value.

        Arguments:

            X (numpy.ndarray): Input data
            derivative (bool): If True, returns the derivative of the tanh function; otherwise, returns the tanh value.
        
        Returns:

            numpy.ndarray: The tanh activation value or its derivative value, depend on the 'derivative' argument

        '''            

        # Calculate tanh activation value
        numerator = np.exp(X) - np.exp(-X)
        denominator = np.exp(X) + np.exp(-X)

        # Return the derivative if requested
        if derivative:
            tan_h = numerator / denominator
            tanh_derivative = 1 - (tan_h ** 2)
            return tanh_derivative
        
        return numerator / denominator
  
    def ReLU(self, X, derivative= False):
        '''
        Transform input data into the Rectified Linear Unit (ReLU) value

        Arguments:

            X (numpy.ndarray): Input data
            derivative (bool): If true, returns the derivative of the Relu function; otherwise, return the Relu value

        Returns:

            numpy.ndarray: The Relu activation value or its derivative value, depend on the 'derivative' argument

        '''

        if derivative:
            return np.where(X > 0, 1, 0)
        return np.maximum(0, X)

    def sigmoid(self, X, derivative= False):
        '''
        Transform input data into the logistic (sigmoid) value

        Arguments:

            X (numpy.ndarray): Input data
            derivative (bool): If true, returns the derivative of the sigmoid function; otherwise, return the sigmoid value

        Returns:

            numpy.ndarray: The sigmoid activation value or its derivative value, depend on the 'derivative' argument

        '''
        # Calculate sigmoid activation value
        sigmoid = 1 / (1 + np.exp(-X))

        # Return its derivative if requested
        if derivative:
            sig = 1 / (1 + np.exp(-X))
            return sig * (1 - sig)
        return sigmoid

    def linear(self, X):
        return X

    def softmax(self, X):
        '''
        Computes the softmax activation for a given input.

        Arguments:
            X (numpy.ndarray): Input array of shape (n,) or (n, m), where n is 
                            the number of classes and m is the number of samples.

        Returns:
            numpy.ndarray: The probabilities of each class, where each value is 
                        in the range (0, 1) and sums to 1 across the specified axis.
        '''
        
        # Subtract the maximum value from X for numerical stability
        exp = np.exp(X - np.max(X))

        # Compute the softmax probabilities by normalizing the exponentiated values
        return exp / np.sum(exp, axis=0, keepdims=True)

    def activate(self, Z, activation, derivative= False):
        '''
        Chooses which activation is chosen based on the input 'activation' argument
        
        Arguments:

            Z (numpy.ndarray): Input data with shape (n, m) this input is fed into activation function
            activation (str): Name of activation 
            derivative (bool): If True, returns the derivative of the activation function; otherwise, returns the activation value

        Returns:

            a function(): A specific activation function based on the name of the input
        '''

        # Return activation function based on its input name
        if activation == 'relu':
            return self.ReLU(Z, derivative= derivative)
        elif activation == 'sigmoid':
            return self.sigmoid(Z, derivative= derivative)
        elif activation == 'tanh':
            return self.tanh(Z, derivative= derivative)
        elif activation == 'softmax':
            return self.softmax(Z)
        elif activation == 'linear':
            return self.linear(Z)
        else:
            # Raise error if user input incorrect activation name
            raise ValueError(f'Unsupported activation function: {activation}')



    def forward(self, X):
        '''
        Performs the feedforward operation through the neural network.

        This function computes the output of the neural network for a given input 
        by applying the weighted sum and activation function at each layer.

        Arguments:
            X (numpy.ndarray): Input data of shape (n, m), where n is the number of features 
                            and m is the number of samples.

        Returns:
            numpy.ndarray: The output of the last layer (activation values) of shape (units, m).
            '''

        # We explicity perform feedforward operation for first layer because the first layer take input data X as input.
        # Perform feedforward operation (Z) for first layer
        self.forward_cache['Z1'] = np.matmul(self.parameters['W1'], X) + self.parameters['b1']
        
        # Apply the activation function (A) to the first layer's output
        self.forward_cache['A1'] = self.activate(self.forward_cache['Z1'], self.activation[1]) # Take the second element from the activation attribute list (0-index)

        # Loop through the remaining layers
        for layer in range(2, self.layers+1):

            # Perform feedforward operation (Z) and apply the activation function (A) for each layer
            self.forward_cache[f'Z{layer}'] = np.matmul(self.parameters[f'W{layer}'], self.forward_cache[f'A{layer-1}']) + self.parameters[f'b{layer}']
            self.forward_cache[f'A{layer}'] = self.activate(self.forward_cache[f'Z{layer}'], self.activation[layer])
    
        # Return the activation output (A) of the last layer
        return self.forward_cache[f'A{self.layers}']



    def categorical_cross_entropy(self, A, y):
        '''
        Computes the categorical cross-entropy loss.

        This function calculates the loss between the predicted probabilities (A) 
        and the true labels (y) for multi-class classification problems.

        Arguments:
            A (numpy.ndarray): Predicted probabilities of shape (num_classes, m), 
                            where m is the number of samples.
            y (numpy.ndarray): True labels that have been one-hot encoded format of shape (num_classes, m).

        Returns:
            float: The computed categorical cross-entropy loss.
        '''

        y_hat = A       # Output from the model
        m = y.shape[1]  # Number of samples
        epsilon = 1e-8  # Small constant to avoid log(0)

        # Compute categorical cross entropy
        loss = (- np.sum(y * np.log(y_hat + epsilon)) / m)
        return loss

    def binary_crossentropy(self, A, y):
        '''
        Computes the binary cross-entropy loss.

        This function calculates the loss between the predicted probabilities (A) 
        and the true labels (y) for binary classification problems.

        Arguments:
            A (numpy.ndarray): Predicted probabilities of shape (1, m), 
                            where m is the number of samples.
            y (numpy.ndarray): True labels of shape (1, m), where m is the number of samples. 
                            Labels should be either 0 or 1.

        Returns:
            float: The computed binary cross-entropy loss.
        '''

        y_hat = A       # Output from the model
        m = y.shape[1]  # Number of samples
        epsilon = 1e-8  # Small constant to avoid log(0)

        # Compute binary cross entropy
        loss = - np.sum(y * np.log(y_hat + epsilon) + (1-y) * np.log(1-y_hat + epsilon)) / m
        return loss

    def mean_square_error(self, A: np.ndarray, y: np.ndarray):
        '''
        Compute the mean square error loss

        This function calculates the loss between the predicted probabilities (A)
        and the true labels (y) for regression problems

        Arguments:
            A (numpy.ndarray): Predicted probabilities of shape (num_classes, m), 
                            where m is the number of samples.
            y (numpy.ndarray): True labels of shape (num_classes, m), where m is the number of samples. 
                            Labels should be either non negative value.
        Returns:
            float: The computed mean square error loss.
        '''
        y_hat = A
        m = y.shape[1]

        loss = (np.sum((y - y_hat)**2) / m)

        return loss
    
    def mean_absolute_error(self, A: np.ndarray, y: np.ndarray):
        '''
        Compute the mean absolute error loss

        This function calculates the loss between the predicted probabilities (A)
        and the true labels (y) for regression problems

        Arguments:
            A (numpy.ndarray): Predicted probabilities of shape (num_classes, m), 
                            where m is the number of samples.
            y (numpy.ndarray): True labels of shape (num_classes, m), where m is the number of samples. 
                            Labels should be either non negative value.
        Returns:
            float: The computed mean absolute error loss.
        '''
        y_hat = A
        m = y.shape[1]

        loss = (np.sum(np.abs(y - y_hat)) / m)
        return loss
    
    def pick_loss(self, losses= None, A= None, y= None):
        '''
        Selects and computes the specified loss function.

        This function retrieves a loss function based on the provided name 
        and computes the loss using the predicted values (A) and true labels (y).
        
        The loss function to use should be previously compiled using the 
        compile() function, which sets the compiled_loss attribute.

        Arguments:
            losses (str): Name of the loss function to compute (e.g., 
                        'binary_crossentropy' or 'categorical_crossentropy').
            A (numpy.ndarray): Predicted values from the model.
            y (numpy.ndarray): True labels corresponding to the predictions.

        Returns:
            float: The computed loss based on the specified loss function.

        Raises:
            ValueError: If the specified loss function name is not recognized 
                        or if no loss function has been compiled.
        '''

        # Define a dictionary mapping loss name to theirs respective functions
        loss_dict = {'binary_crossentropy': self.binary_crossentropy,
                        'categorical_crossentropy': self.categorical_cross_entropy,
                        'mse': self.mean_square_error,
                        'mae': self.mean_absolute_error}

        # Check if a loss function has been compiled;  raise an error if not
        if self.compiled_loss is not None:
            
            # If the specified loss name is in the dictionary, compute and return the loss; raise another error if its not in dictionary
            if losses in loss_dict:
                return loss_dict[losses](A, y)
            
            else:
                raise ValueError(f'No such loss name {losses}')
            
        else:
            raise ValueError(f'Loss has not been compiled')



    def gradients(self, A: np.ndarray, X: np.ndarray, y: np.ndarray, layer: int = 0, output_layer: bool=False, first_layer: bool=False, losses: str= None):
        '''
        Compute the gradients for corresponding layer in the model. 

        This function calculates the gradient of an layer. The gradients is later used again in backpropagation function to update weights and bias

        Arguments:
            A (numpy.ndarray): The activations (predicted values) from the previous layer or from the model.
            X (numpy.ndarray): The input features or activations for the current layer.
            y (numpy.ndarray): The true labels corresponding to the model's predictions.
            layer (int): The index of the layer for which the gradient is being computed (default is 0).
            output_layer (bool): Whether the current layer is the output layer. Set to True if computing gradients for the output layer (default is False).
            first_layer (bool): Whether the current layer is the first layer in the network. Set to True for the first layer (default is False).
            losses (str): The loss function used for computing the gradients (e.g., 'mse', 'cross_entropy'). This is optional and can be None (default is None).

        Returns:
            None: This function does not return a value, but updates internal model gradients.
        '''
        m = y.shape[1]  # Number of examples
        clip = 1  # Gradient clipping value

        # Calculate gradients for the output layer
        # Because the output layer gradient depend on which activations function being used in feedforward, so we create a condition to check for the activation function
        if output_layer and not first_layer:
            
            if losses == 'binary_crossentropy':
                self.backward_cache[f'dZ{layer}'] = (A - y) * A * (1 - A)
            elif losses == 'mse':
                self.backward_cache[f'dZ{layer}'] = (2 * (A - y)) / m
            elif losses == 'mae':
                self.backward_cache[f'dZ{layer}'] = np.where(A > y, 1, -1) / m
            else:
                self.backward_cache[f'dZ{layer}'] = A - y
            
            # Compute dW and db for the output layer
            self.backward_cache[f'dW{layer}'] = (np.matmul(self.backward_cache[f'dZ{layer}'], self.forward_cache[f'A{layer-1}'].T)) / m
            self.backward_cache[f'db{layer}'] = (np.sum(self.backward_cache[f'dZ{layer}'], axis=1, keepdims=True)) / m

        # Because the first layer take X as input, so we can't loop through layers
        # Calculate gradients for the first layer
        elif first_layer and not output_layer:

            self.backward_cache[f'dZ{layer}'] = ( np.matmul(self.parameters[f'W{layer+1}'].T, self.backward_cache[f'dZ{layer+1}']) *
                                                  self.activate(self.forward_cache[f'Z{layer}'], activation=self.activation[layer], derivative=True))
            self.backward_cache[f'dW{layer}'] = np.matmul(self.backward_cache[f'dZ{layer}'], X.T) / m
            self.backward_cache[f'db{layer}'] = np.sum(self.backward_cache[f'dZ{layer}'], axis=1, keepdims=True) / m

        # The hidden layers are similar so we can apply the same logic for all layers 
        # Calculate gradients for hidden layers
        else:

            self.backward_cache[f'dZ{layer}'] = ( np.matmul(self.parameters[f'W{layer+1}'].T, self.backward_cache[f'dZ{layer+1}']) * 
                                                 self.activate(self.forward_cache[f'Z{layer}'], activation=self.activation[layer], derivative=True))
            self.backward_cache[f'dW{layer}'] = np.matmul(self.backward_cache[f'dZ{layer}'], self.forward_cache[f'A{layer-1}'].T) / m
            self.backward_cache[f'db{layer}'] = np.sum(self.backward_cache[f'dZ{layer}'], axis=1, keepdims=True) / m

        # Apply clipping to the gradient to prevent it explode
        self.backward_cache[f'dW{layer}'] = np.clip(self.backward_cache[f'dW{layer}'], -clip, clip)
        self.backward_cache[f'db{layer}'] = np.clip(self.backward_cache[f'db{layer}'], -clip, clip)


    def backpropagation(self, A, X, y, losses= None):
        '''
        Perform backpropagation for all layers in the model.

        This function iterates through the layers in reverse order and calculates 
        parameter gradients for each layer. Depending on the position of the layer 
        (first, last, or hidden), it adjusts the gradient computation accordingly. 

        Parameters:
            A (numpy.ndarray):
                The output prediction from the last layer
            X (numpy.ndarray):
                The input data to the network.
            y (numpy.ndarray):
                The target values for the input data.
            losses (str):
                A string to specify the loss function use in the model. Used to determine the way to calculate gradients for last layer
        
        Notes:
            This function calls the `gradients` method, passing in:
            - `output_layer=True` if the current layer is the output layer,
            - `first_layer=True` if the current layer is the first layer,
            - `output_layer=False` and `first_layer=False` for hidden layers.
        '''

        # Iterate through the layers in reverse order and check if layer is the first, last, or a hidden layer
        for layer in reversed(range(1, self.layers+1)):
    
            if layer == self.layers:
                self.gradients(A, X, y, layer= layer, output_layer= True, first_layer= False, losses= losses)
            elif layer == 1:
                self.gradients(A, X, y, layer= layer, output_layer= False, first_layer= True, losses= losses)
            else:
                self.gradients(A, X, y, layer= layer, output_layer= False, first_layer= False, losses= losses)



    def initialization(self, initializers= 'constant', shape= None, n_in= None, n_out= None):
        '''
        Initialize weights based on the specified initializer method.

        This function provides different weight initialization strategies for 
        layers in a neural network, such as constant, He, and Glorot initializations.
        Each initializer has specific characteristics to help with the convergence 
        and stability of the training process.

        Parameters:
            initializers (str, default= 'constant'):
                The type of initializer to use. Options are:
                - 'constant': returns a constant value of 1.
                - 'he_normal': He normal initialization 
                - 'he_uniform': He uniform initialization 
                - 'glorot_normal': Glorot normal initialization 
                - 'glorot_uniform': Glorot uniform initialization.
            shape (tuple):
                Shape of the weight matrix (required for uniform initializers).
            n_in (int):
                Number of input units to the layer (used in He and Glorot initializers).
            n_out (int):
                Number of output units from the layer (used in Glorot initializers).

        Returns:
            np.ndarray or int:
                The initialized weights based on the chosen method. Returns a single 
                constant value if 'constant' initializer is chosen, otherwise returns 
                a weight matrix.

        Notes:
            This function leverages numpy for uniform random sampling and requires 
            `n_in` and `n_out` values for methods based on input-output layer sizes.
        '''
        # Compute and return the regularization term if it meet the condition

        if initializers == 'constant':
            return 1

        elif initializers == 'he_normal':
            return np.sqrt(2./n_in)

        elif initializers == 'he_uniform':
            limit = np.sqrt(6./n_in)
            return np.random.uniform(low= -limit, high= limit, size= shape)

        elif initializers == 'glorot_normal':
            return np.sqrt(2./(n_in + n_out))

        elif initializers == 'glorot_uniform':
            limit = np.sqrt(6./(n_in + n_out))
            return np.random.uniform(low= -limit, high= limit, size= shape)



    def l1_regularizer(self, lamda= 1, layer= None):
        '''
        This function compute L1 (Lasso) regularization 

        Arguments:
            lamda (float): 
                The regularization hyperparameter controlling the strength of the penalty 
                applied to the weights. Must be a non-negative value. Default is 0.
            layer (int):
                The layer number for which the regularization is computed.
        Returns:
            numpy.ndarray:
                An array with the L1 regularization term for the weights of the specified layer.
        '''
        return lamda * np.sign(self.parameters[f'W{layer}'])
    
    def l2_regularizer(self, lamda= 1, layer= None):
        '''
        This function compute L2 (Ridge) regularization 

        Arguments:
            lamda (float): 
                The regularization hyperparameter controlling the strength of the penalty 
                applied to the weights. Must be a non-negative value. Default is 0.
            layer (int):
                The layer number for which the regularization is computed.
        Returns:
            numpy.ndarray:
                An array with the L2 regularization term for the weights of the specified layer.
        '''

        return 2 * lamda * self.parameters[f'W{layer}']
    
    def pick_regularizers(self, layer= None):
        '''
        Select and apply the specified regularizer for a given layer.

        This function retrieves the regularizer type and hyperparameter (`lamda`) 
        for the specified layer and applies the corresponding regularization 
        method (either L1 or L2). If no regularizer is specified, it returns 0.

        Parameters:
            layer (int):
                The layer number for which the regularizer is to be applied.

        Returns:
            np.ndarray or int:
                The regularization term for the specified layer's weights:
                - If the regularizer is 'l1', returns the result of `l1_regularizer`.
                - If the regularizer is 'l2', returns the result of `l2_regularizer`.
                - If no regularizer is specified, returns 0.

        Notes:
        ------
        This function requires that the `regularizers` attribute contains a dictionary 
        mapping each layer to a tuple or a list (`regularizer`, `lamda`). The specified 
        regularization method is called based on the value of `regularizer`.

        This function will be called when updating weights parameter
        '''
        
        # Retrieve regularizer name and lambda value from regularizers class attribute
        regularizer, lamda = self.regularizers[layer]
        
        # Return the regularization value for the specified layer based on the chosen regularizer.
        if regularizer == 'l1':
            return self.l1_regularizer(lamda= lamda, layer= layer)
        elif regularizer == 'l2':
            return self.l2_regularizer(lamda= lamda, layer= layer)
        else:
            return 0



    def momentum(self, gamma= 0.9, learning_rate= 0.001, layer= None,):
        '''
        Apply Momentum optimization for a specified layer.

        This function implements momentum-based gradient descent, where the update 
        for each parameter is based on an exponentially weighted average of past gradients. 
        Momentum helps accelerate gradients vectors in the right direction, improving 
        convergence.

        Parameters:
            gamma (float):
                The momentum hyperparameter that controls the exponential decay rate of 
                past gradients. Default is 0.9.
            learning_rate (float):
                The step size for each update. Default is 0.001.
            layer (int):
                The layer number to which the momentum update is applied.
        
        Notes:
        ------
            This function initializes velocity terms (`v_w`, `v_b`)
        '''
        # Retrieve weight dimension
        n, m = self.parameters[f'W{layer}'].shape


        # Initialize velocity terms (`v_w`, `v_b`) to the matrix of 0. if they haven't been initialized
        if f'v_w{layer}' not in self.velocity:
            self.velocity[f'v_w{layer}'] = np.zeros((n, m))
        if f'v_b{layer}' not in self.velocity:
            self.velocity[f'v_b{layer}'] = np.zeros((n, 1))


        # Compute velocity terms (`v_w`, `v_b`) for a current layer 
        self.velocity[f'v_w{layer}'] = gamma * self.velocity[f'v_w{layer}'] + (1-gamma) * self.backward_cache[f'dW{layer}']
        self.velocity[f'v_b{layer}'] = gamma * self.velocity[f'v_b{layer}'] + (1-gamma) * self.backward_cache[f'db{layer}']

        # Using the velocity temrs to update weights and biases, and also adding the regularization term to the weights
        self.parameters[f'W{layer}'] -= learning_rate * (self.velocity[f'v_w{layer}'] + self.pick_regularizers(layer= layer))
        self.parameters[f'b{layer}'] -= learning_rate * self.velocity[f'v_b{layer}']


    def gradient_descent(self, learning_rate= 0.001, layer=None,):
        '''
        Apply standard Gradient Descent optimization for a specified layer.

        This function performs a simple gradient descent update, where each parameter 
        is adjusted by moving in the direction of the negative gradient of the loss 
        function with respect to that parameter.

        Parameters:
            learning_rate(float, optional):
                The step size for each update. Default is 0.001.
            layer(int):
                The layer number to which the gradient descent update is applied.

        Notes:
        ------
        This function directly updates weights (`W{layer}`) and biases (`b{layer}`) 
        using gradients from `backward_cache`, and applies the chosen regularizer if 
        specified.
        '''

        # Updating weights and biases using their respective gradients term and also adding the regularization term to the weights

        self.parameters[f'W{layer}'] -= learning_rate * (self.backward_cache[f'dW{layer}'] + self.pick_regularizers(layer= layer))
        self.parameters[f'b{layer}'] -= learning_rate * self.backward_cache[f'db{layer}']


    def RMSprop(self, beta= 0.9, learning_rate= 0.001, epsilon= 1e-8, layer= None,):
        '''
        Apply RMSprop optimization for a specified layer.

        RMSprop (Root Mean Square Propagation) adjusts the learning rate of each 
        parameter based on the magnitude of recent gradients. This helps to mitigate 
        oscillations in the learning path, leading to faster convergence.

        Parameters:
            beta(float, optional):
                The exponential decay rate for the moving average of squared gradients. 
                Default is 0.9.
            learning_rate (float, optional):
                The step size for each update. Default is 0.001.
            epsilon (float, optional):
                A small constant added for numerical stability. Default is 1e-8.
            layer (int):
                The layer number to which the RMSprop update is applied.

        Notes:
        ------
        This function initializes exponential weighted moving averages (`e_w`, 
        `e_b`)
        '''
        
        # Retrieving the weight dimension
        n, m = self.parameters[f'W{layer}'].shape
        

        # Initializing `e_w and `e_b` terms (exponential weighted moving average) to the matrix of 0. if they haven't been initialized
        if f'e_w{layer}' not in self.EWMA:
            self.EWMA[f'e_w{layer}'] = np.zeros((n, m))
        if f'e_b{layer}' not in self.EWMA:
            self.EWMA[f'e_b{layer}'] = np.zeros((n, 1))

        # Compute the exponential weighted moving average terms (`e_w`, `e_b`) for a current layer 
        self.EWMA[f'e_w{layer}'] = (beta * self.EWMA[f'e_w{layer}']) + ((1-beta) * np.power(self.backward_cache[f'dW{layer}'], 2))
        self.EWMA[f'e_b{layer}'] = (beta * self.EWMA[f'e_b{layer}']) + ((1-beta) * np.power(self.backward_cache[f'db{layer}'], 2))

        # Updating weights and biases using their respective gradients term and also adding the regularization term to the weights
        self.parameters[f'W{layer}'] -= learning_rate * (((self.backward_cache[f'dW{layer}']) / (np.sqrt(self.EWMA[f'e_w{layer}']) + epsilon)) + self.pick_regularizers(layer= layer))
        self.parameters[f'b{layer}'] -= learning_rate * self.backward_cache[f'db{layer}'] / (np.sqrt(self.EWMA[f'e_b{layer}']) + epsilon)
                                                                                        

    def Adam(self, learning_rate= 0.001, beta1= 0.9, beta2= 0.999, epsilon= 1e-8, layer= None,):
        '''
        Apply Adam optimization for a specified layer.

        Adam (Adaptive Moment Estimation) combines the advantages of both Momentum 
        and RMSprop. It computes adaptive learning rates for each parameter by 
        maintaining running averages of both the gradients and their squared values.

        Parameters:
            learning_rate (float, optional):
                The step size for each update. Default is 0.001.
            beta1 (float, optional):
                The exponential decay rate for the first moment estimates (mean of gradients). 
                Default is 0.9.
            beta2 (float, optional):
                The exponential decay rate for the second moment estimates (squared gradients). 
                Default is 0.999.
            epsilon (float, optional):
                A small constant added for numerical stability. Default is 1e-8.
            layer (int):
                The layer number to which the Adam update is applied.

        Notes:
        ------
        This function initializes moment terms (`v_w`, `v_b, `s_w`,`s_b`) 
        if not already present in the "moments" dictionary. Adam updates 
        use bias-corrected estimates and apply regularization if specified.
        '''    

        # Retrieving the weight dimension and increasing time step t by 1
        n, m = self.parameters[f'W{layer}'].shape
        self.t += 1

        # Initializing `v_w and `v_b` terms (first moment) to the matrix of 0. if they haven't been initialized
        if f'v_w{layer}' not in self.moments or f'v_b{layer}' not in self.moments:
            self.moments[f'v_w{layer}'] = np.zeros((n, m))
            self.moments[f'v_b{layer}'] = np.zeros((n, 1))

        # Initializing `s_w and `s_b` terms (second moment) to the matrix of 0. if they haven't been initialized
        if f's_w{layer}' not in self.moments or f's_b{layer}' not in self.moments:
            self.moments[f's_w{layer}'] = np.zeros((n, m))
            self.moments[f's_b{layer}'] = np.zeros((n, 1))

        # Compute the first and second moment terms (`v_w`, `v_b`, `s_w``, `s_b`) for a current layer 
        self.moments[f'v_w{layer}'] = beta1 * self.moments[f'v_w{layer}'] + (1-beta1) * self.backward_cache[f'dW{layer}']
        self.moments[f'v_b{layer}'] = beta1 * self.moments[f'v_b{layer}'] + (1-beta1) * self.backward_cache[f'db{layer}']
        self.moments[f's_w{layer}'] = beta2 * self.moments[f's_w{layer}'] + (1-beta2) * (self.backward_cache[f'dW{layer}']**2)
        self.moments[f's_b{layer}'] = beta2 * self.moments[f's_b{layer}'] + (1-beta2) * (self.backward_cache[f'db{layer}']**2)

        # Compute correct bias to avoid skewing the results toward zero
        self.moments[f'v_w_cb{layer}'] = self.moments[f'v_w{layer}'] / (1-(beta1**self.t))
        self.moments[f'v_b_cb{layer}'] = self.moments[f'v_b{layer}'] / (1-(beta1**self.t))
        self.moments[f's_w_cb{layer}'] = self.moments[f's_w{layer}'] / (1-(beta2**self.t))
        self.moments[f's_b_cb{layer}'] = self.moments[f's_b{layer}'] / (1-(beta2**self.t))

        # Updating weights and biases using their respective gradients term and also adding the regularization term to the weights
        self.parameters[f'W{layer}'] -= learning_rate * (((self.moments[f'v_w_cb{layer}']) / (np.sqrt(self.moments[f's_w_cb{layer}']) + epsilon)) + self.pick_regularizers(layer= layer))
        self.parameters[f'b{layer}'] -= learning_rate * self.moments[f'v_b_cb{layer}'] / (np.sqrt(self.moments[f's_b_cb{layer}']) + epsilon)


    def pick_optimizer(self, optimizer= None, layer= None, **kwargs):
        """
        Selects and applies the optimizer for a specified layer.

        Parameters:
            optimizer (str): The name of the optimizer to use ('momentum', 'sgd', 'adam', 'rmsprop').
            layer (int): The layer on which to apply the optimizer.
            **kwargs: Additional keyword arguments for optimizer-specific parameters.

        Raises:
            ValueError: If an unknown optimizer is specified.
        """
        # Create a dictionary contain key name of optimizers mapping to their corresponding function
        optimizer_dict = {'momentum': self.momentum,
                            'sgd': self.gradient_descent,
                            'adam': self.Adam,
                            'rmsprop': self.RMSprop,}
    
        # Check if the optimizer name is in the dictionary. If it is, call the function with respect to the name of the optimizer and its arguments.
        if optimizer in optimizer_dict:
            optimizer_dict[optimizer](layer= layer, **kwargs)
        else:
            raise ValueError(f'Unknown optimizer: "{optimizer}"')
        

    def update_parameters(self):
        """
        Updates the parameters of the model using the compiled optimizer.

        Raises:
            ValueError: If no optimizer has been compiled.
        """
        # Raise an error if user forget to compile optimizer name
        if self.compiled_optimizer is None:
            raise ValueError('Optimizer has not been compiled')
        
        # Loop through layers in reverse order
        for l in reversed(range(1, self.layers+1)):
            
            # Retrieving optimizer name and its hyperparameters 
            optimizer_name = self.compiled_optimizer['name']
            optimizer_params = self.compiled_optimizer['params']
            
            # Call the pick_optimizer() method to update parameters
            self.pick_optimizer(optimizer_name, layer= l, **optimizer_params)



    # We have an boolean argument _train here, because when training the model, to print out the accuracy at each epoch we have to transform the prediction
    # to match the dimension of true label.
    def accuracy(self, y_pred, y,  _train= False):
        """
        Computes the accuracy of predictions.

        Parameters:
            y_pred (ndarray): Predicted labels or probabilities.
            y (ndarray): True labels.
            _train (bool): Whether the model is in training mode, requiring dimension adjustment for predictions. (User do not change this)

        Returns:
            float: The accuracy score.
        """
        if _train:
            y_pred = np.argmax(y_pred, axis =0)

        # Convert true labels to class labels (assuming y is one-hot encoded)
        y = np.argmax(y, axis=0) 
        return np.mean(y_pred == y)


    def precision(self, y_pred, y, _train=False):  
        """
        Computes the precision score for each class.

        Parameters:
            y_pred (ndarray): Predicted labels or probabilities.
            y (ndarray): True labels.
            _train (bool): Whether the model is in training mode, requiring dimension adjustment for predictions. (User do not change this)

        Returns:
            float: The mean precision score across classes.
        """      
        if _train:
            y_pred = np.argmax(y_pred, axis=0)

        # Get unique class labels from the true labels
        classes = np.unique(y)
        precision_scores = []  # Initialize a list to store precision scores for each class

        # Convert true labels to class labels (assuming y is one-hot encoded)
        y = np.argmax(y, axis=0)
        
        # Calculate precision for each class
        for cls in classes:
            # Calculate true positives (TP): correct predictions for the class
            tp = np.sum((y_pred == cls) & (y == cls))
            # Calculate false positives (FP): incorrect predictions for the class
            fp = np.sum((y_pred == cls) & (y != cls))
            
            # Calculate precision: TP / (TP + FP) 
            # If there are no positive predictions (TP + FP = 0), set precision to 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_scores.append(precision)  # Append the precision score for the class
        
        return np.mean(precision_scores)


    def recall(self, y_pred, y,   _train= False):
        """
        Computes the recall score for each class.

        Parameters:
            y_pred (ndarray): Predicted labels or probabilities.
            y (ndarray): True labels.
            _train (bool): Whether the model is in training mode, requiring dimension adjustment for predictions. (User do not change this)

        Returns:
            float: The mean recall score across classes.
        """
        
        if _train:
            y_pred = np.argmax(y_pred, axis =0)    

        # Get unique class labels from the true labels
        classes = np.unique(y)
        recall_scores= [] # Initialize a list to store precision scores for each class

        # Convert true labels to class labels (assuming y is one-hot encoded)
        y = np.argmax(y, axis=0) 

        # Calculate precision for each class
        for cls in classes:

            # Calculate true positives (TP): correct predictions for the class
            tp = np.sum((y_pred == cls) & (y == cls))
            # Calculate false negatives (FN): incorrect predictions for the class  
            fn = np.sum((y_pred != cls) & (y == cls))

            # Calculate precision: TP / (TP + Fn) 
            # If there are no positive predictions (TP + FP = 0), set precision to 0
            recall = tp / (tp + fn) if (tp + fn ) > 0 else 0
            recall_scores.append(recall)  # Append the precision score for the class

        return np.mean(recall_scores)


    def f1_score(self, y_pred, y,  _train=False):
        """
        Computes the F1 score, which is the harmonic mean of precision and recall.

        Parameters:
            y_pred (ndarray): Predicted labels or probabilities.
            y (ndarray): True labels.
            _train (bool): Whether the model is in training mode, requiring dimension adjustment for predictions. (User do not change this)

        Returns:
            float: The F1 score.
        """

        # Calculate precision and recall by calling the precision and recall method
        precision = self.precision(y_pred, y, _train=_train)
        recall = self.recall(y_pred, y, _train=_train)
        
        # Compute the F1 score using the formula:
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        # Handle division by zero by checking if precision + recall is greater than 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1  


    def pick_metric(self, metrics= None, y_pred= None, y= None, _train= False):
        """
        Selects and calculates the specified evaluation metric.

        Parameters:
            metrics (str): Name of the metric ('accuracy', 'precision', 'recall', 'f1').
            y_pred (ndarray): Predicted labels or probabilities.
            y (ndarray): True labels.
            _train (bool): Whether the model is in training mode. (User do not change this.)

        Returns:
            float: The computed metric value.

        Raises:
            ValueError: If no metric has been compiled or the specified metric is not valid.
        """
        # Dictionary to map metric names to their respective methods
        metric_dict = {'accuracy': self.accuracy,
                        'precision': self.precision,
                        'recall': self.recall,
                        'f1': self.f1_score}
    
        # Check if a compiled metric exists
        if self.compiled_metric is not None:

            # Check if the requested metric is in the dictionary of available metrics then call its function
            if metrics in metric_dict:
                return metric_dict[metrics](y_pred= y_pred, y= y, _train= _train)
            
            else:
                # Raise an error if the metric name is not valid
                raise ValueError(f'No such metric name {metrics}')
        else:
            # Raise an error if no metric has been compiled
            raise ValueError(f'Metric has not been compiled')



    def compile(self, optimizer=None, losses=None, metrics=None):
        """
        Compiles the model with specified optimizer, loss, and metric.

        Parameters:
            optimizer (str or tuple): Optimizer name or a tuple (name, hyperparameters).
            losses (str): Loss function name.
            metrics (str): Metric name.

        Raises:
            ValueError: If the optimizer format is invalid or not recognized.
        """

        # Dictionary to map optimizer names to their respective hyperparameters
        opt_params = {'sgd': {'learning_rate': 0.01},
                        'momentum': {'learning_rate': 0.01, 'gamma': 0.9},
                        'adam': {'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8},
                        'rmsprop': {'learning_rate': 0.001, 'beta': 0.9, 'epsilon': 1e-8,},}
        
        # If loss function name is a string assign it to class compiled_loss attribute
        if isinstance(losses, str): 
            self.compiled_loss = losses

        # If metric name is a string assign it to class compiled_metric attribute
        if isinstance(metrics, str):
            self.compiled_metric = metrics

        # If optimizer function name is a string assign it to class compiled_optimizer attribute
        if isinstance(optimizer, str):
            if optimizer in opt_params:
                self.compiled_optimizer = {'name': optimizer, 'params': opt_params[optimizer]}
            else:
                raise ValueError(f'No such optimizer name: {optimizer}')
    
        # If optimizer is a list or a tuple contain 2 elements: name (str), hyperparameter (dict)
        elif isinstance(optimizer, (list, tuple)) and len(optimizer) == 2:
            optimizer_name, optimizer_params = optimizer    # Retrieve optimizer name and optimizer hyperparameter
            
            # Another condition to check if optimizer_params is a dictionary
            if isinstance(optimizer_params, dict):
                self.compiled_optimizer = {'name': optimizer_name, 'params': optimizer_params} #
            else:
                # Raise an error if the hyperparameters are not in dictionary format
                raise ValueError("The second element must be a dictionary!")
            
        # Raise an error if the optimizer format is invalid
        else:
            raise ValueError(f'Invalid optimizer format: {optimizer}')
        


    def fit(self, X, y, batch_size=1, epochs=1, verbose=False, patience=float('inf')):
        """
        Trains the model for a specified number of epochs with optional early stopping.

        Parameters:
            X (ndarray): Training input data.
            y (ndarray): True labels for the training data.
            batch_size (int): Number of samples per batch.
            epochs (int): Total number of epochs for training.
            verbose (bool): If True, prints progress information.
            patience (int): Number of epochs without improvement before stopping.

        Returns:
            dict: History of loss and metric values across epochs.
        """
        m = X.shape[1]  # Number of samples
        previous_loss = float('inf')  # Initialize the previous loss for early stopping

        # History to track loss and metric values across epochs
        history = {'metric': [], 'loss': []}
        patience_counter = 0  # Counter for early stopping

        # Main training loop for specified number of epochs
        for epoch in range(1, epochs + 1):
            # Shuffle the dataset for each epoch
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]
            epoch_loss = 0

            # Iterate through batches of data
            for batch in range(0, m, batch_size):
                # Create batches of data
                X_batches = X_shuffled[:, batch:batch + batch_size]
                y_batches = y_shuffled[:, batch:batch + batch_size]

                # Forward pass: compute output
                A = self.forward(X_batches)

                # Backpropagation: update model parameters based on loss
                self.backpropagation(A, X_batches, y_batches, losses=self.compiled_loss)
                self.update_parameters()  # Update model parameters

            # Evaluate the model on the full dataset
            y_pred = self.forward(X)
            epoch_loss = self.pick_loss(losses=self.compiled_loss, A=y_pred, y=y)  # Calculate loss
            metric = self.pick_metric(metrics=self.compiled_metric, y_pred=y_pred, y=y, _train=True)  # Calculate metric
            history['loss'].append(epoch_loss)  # Record loss
            history['metric'].append(metric)  # Record metric

            # Verbose output for monitoring
            if verbose:
                print(f'Cost at epoch {epoch}: {epoch_loss}')
                print(f"Epoch {epoch}/{epochs} - {self.compiled_metric}: {metric * 100:.2f}%\n")             

            # Early stopping logic
            if epoch_loss < previous_loss:
                self.best_parameters = copy.deepcopy(self.parameters)  # Save best parameters
                previous_loss = epoch_loss  # Update previous loss
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1  # Increment patience counter
                if patience < patience_counter:  # Check for early stopping condition
                    print(f'Early Stopping at epoch: {epoch}')
                    break

        # Clear caches and parameters to free memory
        self.backward_cache.clear()
        self.parameters.clear()
        self.velocity.clear()
        self.EWMA.clear()
        self.moments.clear()

        return history  # Return history of metrics and losses


    def predict(self, X):      
        """
        Generates predictions based on input data.

        Parameters:
            X (ndarray): Input data for prediction.

        Returns:
            ndarray: Predicted class labels.
        """  
        A = X
        # Loop through layers
        for layer in range(1, self.layers+1):

            # Perform feedforward operation (Z) and apply the activation function (A) for each layer
            Z = np.matmul(self.best_parameters[f'W{layer}'], A) + self.best_parameters[f'b{layer}']
            A = self.activate(Z, self.activation[layer])
    
        # Return the activation output (A) of the last layer
        return np.argmax(A, axis= 0)

    
    
    def best_parameters(self):
        return self.best_parameters


    def save_model(self, filepath):
        np.savez(filepath, 
                layers=self.layers, 
                activation=self.activation, 
                **self.best_parameters)


    def load_model(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        
        self.layers = int(data['layers'])
        self.activation = list(data['activation'])
        self.best_parameters = {key: data[key] for key in data if key not in ['layers', 'activation']}

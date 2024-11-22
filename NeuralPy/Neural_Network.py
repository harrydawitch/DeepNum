import numpy as np
import copy

class DNN():

    def __init__(self):
        self.forward_cache = {}     
        self.backward_cache = {} 
        self.parameters= {}
        self.best_parameters= None
        self.dropout_mask = {}

        self.layers = 0
        self.dropout_layers = []
        self.activation = [0]
        self.regularizers = [0]

        self.compiled_optimizer = None
        self.compiled_loss = None
        self.compiled_metric = None

        self.velocity = {}
        self.EWMA= {}
        self.moments= {}
        self.t = 0
    
    def Dropout(self, rate= 0):

        if self.layers < 1:
            raise Exception('You must create at least one layer before adding a dropout layer')

        self.dropout_layers.append(self.layers)
        self.dropout_mask[f'p{self.layers}'] = rate


    def apply_dropout(self, layer):
        n, m = self.forward_cache[f'A{layer}'].shape
            
        keep_prob = self.dropout_mask[f'p{layer}']
        mask = np.random.rand(n, m) < keep_prob
        
        self.dropout_mask[f'D{layer}'] = mask
        
        return mask

    def Dense(self, units= None, input_dim= None, activation= 'linear', initializers= 'constant', regularizer= (None, 0)):

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

        if derivative:
            return np.where(X > 0, 1, 0)
        return np.maximum(0, X)

    def sigmoid(self, X, derivative= False):

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
        exp = np.exp(X - np.max(X, axis=0, keepdims=True))
        return exp / (np.sum(exp, axis=0, keepdims=True) + 1e-8)


    def activate(self, Z, activation, derivative= False):
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
        # Loop through the layers
        for layer in range(1, self.layers+1):

            if layer == 1:

                self.forward_cache[f'Z{layer}'] = np.matmul(self.parameters[f'W{layer}'], X) + self.parameters[f'b{layer}']
                self.forward_cache[f'A{layer}'] = self.activate(self.forward_cache[f'Z{layer}'], self.activation[layer])
            
            else:
                # Perform feedforward operation (Z) and apply the activation function (A) for each layer
                self.forward_cache[f'Z{layer}'] = np.matmul(self.parameters[f'W{layer}'], self.forward_cache[f'A{layer-1}']) + self.parameters[f'b{layer}']
                self.forward_cache[f'A{layer}'] = self.activate(self.forward_cache[f'Z{layer}'], self.activation[layer])

            if  layer in self.dropout_layers:
                mask = self.apply_dropout(layer= layer)
                keep_prob = self.dropout_mask[f'p{layer}']

                self.forward_cache[f'A{layer}'] = self.forward_cache[f'A{layer}'] * mask / keep_prob
    
        # Return the activation output (A) of the last layer
        return self.forward_cache[f'A{self.layers}']



    def categorical_cross_entropy(self, A, y):

        y_hat = A       # Output from the model
        m = y.shape[1]  # Number of samples
        epsilon = 1e-8  # Small constant to avoid log(0)

        # Compute categorical cross entropy
        loss = (- np.sum(y * np.log(y_hat + epsilon)) / m)
        return loss

    def binary_crossentropy(self, A, y):

        y_hat = A       # Output from the model
        m = y.shape[1]  # Number of samples
        epsilon = 1e-8  # Small constant to avoid log(0)

        # Compute binary cross entropy
        loss = - np.sum(y * np.log(y_hat + epsilon) + (1-y) * np.log(1-y_hat + epsilon)) / m
        return loss

    def mean_square_error(self, A: np.ndarray, y: np.ndarray):
        y_hat = A
        m = y.shape[1]

        loss = (np.sum((y - y_hat)**2) / m)

        return loss
    
    def mean_absolute_error(self, A: np.ndarray, y: np.ndarray):
        y_hat = A
        m = y.shape[1]

        loss = (np.sum(np.abs(y - y_hat)) / m)
        return loss
    
    def pick_loss(self, losses= None, A= None, y= None):

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
        m = y.shape[1]  # Number of examples
        clip = 1  # Gradient clipping value

        # Calculate gradients for the output layer
        # Because the output layer gradient depend on which activations function being used in feedforward, so we create a condition to check for the activation function
        if output_layer:
            
            if losses == 'binary_crossentropy':
                self.backward_cache[f'dZ{layer}'] = (A - y) * A * (1 - A)
            elif losses == 'mse':
                self.backward_cache[f'dZ{layer}'] = (2 * (A - y)) / m
            elif losses == 'mae':
                self.backward_cache[f'dZ{layer}'] = np.where(A > y, 1, -1) / m
            else:
                self.backward_cache[f'dZ{layer}'] = A - y
            

            if layer in self.dropout_layers:
                self.backward_cache[f'dZ{layer}'] = self.backward_cache[f'dZ{layer}'] * self.dropout_mask[f'D{layer}'] / (self.dropout_mask[f'p{layer}'])
            
            # Compute dW and db for the output layer
            self.backward_cache[f'dW{layer}'] = (np.matmul(self.backward_cache[f'dZ{layer}'], self.forward_cache[f'A{layer-1}'].T)) / m
            self.backward_cache[f'db{layer}'] = (np.sum(self.backward_cache[f'dZ{layer}'], axis=1, keepdims=True)) / m

        # Because the first layer take X as input, so we can't loop through layers
        # Calculate gradients for the first layer
        elif first_layer:

            self.backward_cache[f'dZ{layer}'] = ( np.matmul(self.parameters[f'W{layer+1}'].T, self.backward_cache[f'dZ{layer+1}']) *
                                                  self.activate(self.forward_cache[f'Z{layer}'], activation=self.activation[layer], derivative=True))
            
            if layer in self.dropout_layers:
                self.backward_cache[f'dZ{layer}'] = self.backward_cache[f'dZ{layer}'] * self.dropout_mask[f'D{layer}'] / (1 - self.dropout_mask[f'p{layer}'])
                

            self.backward_cache[f'dW{layer}'] = np.matmul(self.backward_cache[f'dZ{layer}'], X.T) / m
            self.backward_cache[f'db{layer}'] = np.sum(self.backward_cache[f'dZ{layer}'], axis=1, keepdims=True) / m

        # The hidden layers are similar so we can apply the same logic for all layers 
        # Calculate gradients for hidden layers
        else:

            self.backward_cache[f'dZ{layer}'] = ( np.matmul(self.parameters[f'W{layer+1}'].T, self.backward_cache[f'dZ{layer+1}']) * 
                                                 self.activate(self.forward_cache[f'Z{layer}'], activation=self.activation[layer], derivative=True))
            
            if layer in self.dropout_layers:
                self.backward_cache[f'dZ{layer}'] = self.backward_cache[f'dZ{layer}'] * self.dropout_mask[f'D{layer}'] / (1 - self.dropout_mask[f'p{layer}'])


            self.backward_cache[f'dW{layer}'] = np.matmul(self.backward_cache[f'dZ{layer}'], self.forward_cache[f'A{layer-1}'].T) / m
            self.backward_cache[f'db{layer}'] = np.sum(self.backward_cache[f'dZ{layer}'], axis=1, keepdims=True) / m

        # Apply clipping to the gradient to prevent it explode
        self.backward_cache[f'dW{layer}'] = np.clip(self.backward_cache[f'dW{layer}'], -clip, clip)
        self.backward_cache[f'db{layer}'] = np.clip(self.backward_cache[f'db{layer}'], -clip, clip)


    def backpropagation(self, A, X, y, losses= None):

        # Iterate through the layers in reverse order and check if layer is the first, last, or a hidden layer
        for layer in reversed(range(1, self.layers+1)):
    
            if layer == self.layers:
                self.gradients(A, X, y, layer= layer, output_layer= True, first_layer= False, losses= losses)
            elif layer == 1:
                self.gradients(A, X, y, layer= layer, output_layer= False, first_layer= True, losses= losses)
            else:
                self.gradients(A, X, y, layer= layer, output_layer= False, first_layer= False, losses= losses)



    def initialization(self, initializers= None, shape= None, n_in= None, n_out= None):
        # Compute and return the regularization term if it meet the condition

        if initializers == 'he_normal':
            return np.random.normal(loc= 0.0, scale= np.sqrt(2./n_in, size= shape))

        elif initializers == 'he_uniform':
            limit = np.sqrt(6./n_in)
            return np.random.uniform(low= -limit, high= limit, size= shape)

        elif initializers == 'glorot_normal':
            return np.random.normal(loc= 0.0, scale= np.sqrt(2./(n_in + n_out)), size= shape)

        elif initializers == 'glorot_uniform':
            limit = np.sqrt(6./(n_in + n_out))
            return np.random.uniform(low= -limit, high= limit, size= shape)



    def l1_regularizer(self, lamda= 1, layer= None):
        return lamda * np.sign(self.parameters[f'W{layer}'])
    
    def l2_regularizer(self, lamda= 1, layer= None):
        return 2 * lamda * self.parameters[f'W{layer}']
    
    def pick_regularizers(self, layer= None):
        
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

        # Updating weights and biases using their respective gradients term and also adding the regularization term to the weights

        self.parameters[f'W{layer}'] -= learning_rate * (self.backward_cache[f'dW{layer}'] + self.pick_regularizers(layer= layer))
        self.parameters[f'b{layer}'] -= learning_rate * self.backward_cache[f'db{layer}']


    def RMSprop(self, beta= 0.9, learning_rate= 0.001, epsilon= 1e-8, layer= None,):
        
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

        if _train:
            y_pred = np.argmax(y_pred, axis =0)

        # Convert true labels to class labels (assuming y is one-hot encoded)
        y = np.argmax(y, axis=0) 
        return np.mean(y_pred == y)


    def precision(self, y_pred, y, _train=False):  

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

        # Calculate precision and recall by calling the precision and recall method
        precision = self.precision(y_pred, y, _train=_train)
        recall = self.recall(y_pred, y, _train=_train)
        
        # Compute the F1 score using the formula:
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        # Handle division by zero by checking if precision + recall is greater than 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1  


    def pick_metric(self, metrics= None, y_pred= None, y= None, _train= False):
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
            y_pred = self.predict(X, training= True)

            epoch_loss = self.pick_loss(losses=self.compiled_loss, A= y_pred, y= y)  # Calculate loss
            metric = self.pick_metric(metrics=self.compiled_metric, y_pred= y_pred, y= y, _train=True)  # Calculate metric
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


    def predict(self, X, training= False):        
        A = X
        
        if training:
            for layer in range(1, self.layers+1):

                Z = np.matmul(self.parameters[f'W{layer}'], A) + self.parameters[f'b{layer}']
                A = self.activate(Z, self.activation[layer])
            return A
        
        else:

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

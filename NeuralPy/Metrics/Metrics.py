import numpy as np
def accuracy(y_pred, y):

    y_pred = np.argmax(y_pred, axis =0)
    y = np.argmax(y, axis =0)

    return np.mean(y_pred == y)

def precision(y_pred, y):  

    # Get unique class labels from the true labels
    classes = np.unique(y)
    precision_scores = []  # Initialize a list to store precision scores for each class

    y_pred = np.argmax(y_pred, axis =0)
    y = np.argmax(y, axis =0)
    
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

def recall(y_pred, y):

    # Get unique class labels from the true labels
    classes = np.unique(y)
    recall_scores= [] # Initialize a list to store precision scores for each class

    # Convert true labels to class labels (assuming y is one-hot encoded)
    y = np.argmax(y, axis=0) 
    y_pred = np.argmax(y_pred, axis =0)    

    # Calculate precision for each class
    for cls in classes:

        # Calculate true positives (TP): correct predictions for the class
        tp = np.sum((y_pred == cls) & (y == cls))
        # Calculate false negatives (FN): incorrect predictions for the class  
        fn = np.sum((y_pred != cls) & (y == cls))

        # Calculate precision: TP / (TP + Fn) 
        # If there are no positive predictions (TP + FP = 0), set precision to 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_scores.append(recall)  # Append the precision score for the class

    return np.mean(recall_scores)

def f1_score(y_pred, y):

    # Calculate precision and recall by calling the precision and recall method
    precisions = precision(y_pred, y)
    recalls = recall(y_pred, y)
    
    # Compute the F1 score using the formula:
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # Handle division by zero by checking if precision + recall is greater than 0
    f1 = (2 * precisions * recalls) / (precisions + recalls) if (precisions + recalls) > 0 else 0
    
    return f1  

def pick_metric(metrics= None, y_pred= None, y_true= None):
    # Dictionary to map metric names to their respective methods
    metric_dict = {'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1_score}

    # Check if a compiled metric exists
    if metrics is not None:

        # Check if the requested metric is in the dictionary of available metrics then call its function
        if metrics in metric_dict:
            return metric_dict[metrics](y_pred= y_pred, y= y_true)
        
        else:
            # Raise an error if the metric name is not valid
            raise ValueError(f'No such metric name {metrics}')
    else:
        # Raise an error if no metric has been compiled
        raise ValueError(f'Metric has not been compiled')
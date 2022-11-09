# Data analysis % wrangling
import pandas as pd
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

#import train and test datasets

train = pd.read_csv(r"C:\Users\COLD\Desktop\dataset\train.csv",names = ['feature_1', 'feature_2', 'feature_3', 'feature_4' , 'classes'])
test = pd.read_csv(r"C:\Users\COLD\Desktop\dataset\test.csv",names = ['feature_1', 'feature_2', 'feature_3', 'feature_4' , 'classes'])

# Split training dataset into binary form (2 classes each dataset)

train_12= train.loc[train['classes'].isin(["class-1","class-2"])]
train_12["classes"] = train_12["classes"].map({"class-1":1, "class-2":-1})
train_12 = train_12.values
np.random.shuffle(train_12)
#---------------------------------------------#
train_23= train.loc[train['classes'].isin(["class-2","class-3"])]
train_23["classes"] = train_23["classes"].map({"class-2":1, "class-3":-1})
train_23 = train_23.values
np.random.shuffle(train_23)
#---------------------------------------------#
train_13= train.loc[train['classes'].isin(["class-1","class-3"])]
train_13["classes"] = train_13["classes"].map({"class-1":1, "class-3":-1})
train_13 = train_13.values
np.random.shuffle(train_13)

#>-------------- Class definition --------------------<

class Perceptron():
    
    def __init__(self, max_iter=30):
        
        self.max_iter = max_iter

    #>-------------- Defining function for training your data --------------------<

    def perceptron_train(self, train_data): 
        # Seprate features and targets
        X = train_data[:, :train_data.shape[1]-1] # Split input data into arrays
        y = train_data[:, -1] # Split targets into arrays
        self.bias = 0
        self.weights = np.zeros(X.shape[1]+1) # set initial weights
        self.errors = [] # store error
        
        for iter in tqdm(range(self.max_iter)): # for each iteration we change all weights
            num_errors = 0 # set initial number of errors before any updates of weights
            
            for xi, yi in zip(X, y): #zip returns an iterator of tuples based on the iterable objects
                a = np.dot(xi,self.weights[1:]) + self.bias #compute activation
                if yi*a <= 0:
                    num_errors += 1
                    self.weights[1:] += self.weights[1:] + yi*xi #update weights
                    self.bias =  self.bias + yi #update bias
                    # self.weights[0] = self.bias
            self.errors.append(num_errors)  #add the number of errors into errors list    
        return self

    def perceptron_test(self, X):
        result = [] # store predicted result (1 or -1)
        for x in X:
            a = np.dot(x,self.weights[1:]) + self.bias # compute activation for test 
            result.append(np.sign(a)) #add the sign of every predicted result into result's list 
        return result

    def accuracy(self,y_pred,y_true):
        # compare y_pred to y_true
        # every time they are not the same error+=1
        errors = 0
        for i,j in zip(y_pred,y_true):
            if i != j:
                errors+=1
        acc = ((len(y_true) - errors) / len(y_true))*100 
        return acc

test_12= test.loc[test['classes'].isin(["class-1","class-2"])]
test_12["classes"] = test_12["classes"].map({"class-1":1, "class-2":-1})
test_12 = test_12.values
X_test_12 = test_12[:, :test.shape[1]-1]
y_true_12 = test_12[:, -1]
#---------------------------------------------#
test_23= test.loc[test['classes'].isin(["class-2","class-3"])]
test_23["classes"] = test_23["classes"].map({"class-2":1, "class-3":-1})
test_23 = test_23.values
X_test_23 = test_23[:, :test.shape[1]-1]
y_true = test_23[:, -1]
#---------------------------------------------#
test_13= test.loc[test['classes'].isin(["class-1","class-3"])]
test_13["classes"] = test_13["classes"].map({"class-1":1, "class-3":-1})
test_13 = test_13.values
X_test_13 = test_13[:, :test.shape[1]-1]
y_true_13 = test_13[:, -1]


num_iter = 2
classifier= Perceptron(num_iter)
classifier.perceptron_train(train_12)
classifier.errors

y_pred_12 = classifier.perceptron_test(X_test_12)
#y_pred_23 = classifier.perceptron_test(X_test_23)
#y_pred_13 = classifier.perceptron_test(X_test_13)


classifier.accuracy(y_pred_12,y_true)
print(f' The accuracy of the model is : {accuracy}%')
# EX03---Implementation-of-MLP-with-Backpropagation

## AIM:
To implement a Multilayer Perceptron for Multi classification.

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook.

## RELATED THEORETICAL CONCEPT:

A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers.A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing.
 
MLP has the following features:

Ø  Adjusts the synaptic weights based on Error Correction Rule

Ø  Adopts LMS

Ø  Consists of two passes

   	(i)  Feed Forward pass
    (ii) Backward pass
           
Ø  Learning process –backpropagation

![image 10](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)

3 Distinctive Characteristics of MLP:

Ø  Each neuron in network includes a non-linear activation function

![image](https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png)

Ø  Contains one or more hidden layers with hidden neurons

Ø  Network exhibits high degree of connectivity determined by the synapses of the network

3 Signals involved in MLP are:

 Functional Signal:

* propagates forward neuron by neuron thro network and emerges at an output signal

* F(x,w) at each neuron as it passes

Error Signal:

   * Originates at an output neuron
   
   * Propagates backward through the network neuron
      

The computation of an estimate of the gradient vector is needed for the backward pass through the network

TWO PASSES OF COMPUTATION:

In the forward pass:

•       Synaptic weights remain unaltered

•       Function signal of jth neuron is
            ![image](https://user-images.githubusercontent.com/112920679/198814313-2426b3a2-5b8f-489e-af0a-674cc85bd89d.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814328-1a69a3cd-7e02-4829-b773-8338ac8dcd35.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814339-9c9e5c30-ac2d-4f50-910c-9732f83cabe4.png)



If jth neuron is output neuron, the m=mL  and output of j th neuron is
               ![image](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)

Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer
![image](https://user-images.githubusercontent.com/112920679/198814353-276eadb5-116e-4941-b04e-e96befae02ed.png)


In the backward pass,

•       It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

![image](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)



## ALGORITHM:

1.Import the necessary libraries of python.


2. After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.


3. Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%.
Normalize our dataset. 


4.Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.


5.In order to get the predicted values we call the predict() function on the testing data set.


6. Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.

## PROGRAM:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("IRIS.csv")
df

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X.head()
y.head()

print(y.unique())
le = LabelEncoder()
y = le.fit_transform(y)
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
mlp.fit(X_train, y_train)  
predictions = mlp.predict(X_test) 

print(predictions)

accuracy_score(y_test,predictions)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
```
## OUTPUT:
### Reading Dataset:
![image](https://user-images.githubusercontent.com/94164665/232534044-350cb220-bc6a-4b03-b135-037225568241.png)
### First five values of X:
![image](https://user-images.githubusercontent.com/94164665/232534303-3bdc8ccb-eace-498e-a009-2ba68bfc5cb5.png)
### First five values of Y:
![image](https://user-images.githubusercontent.com/94164665/232534489-59154c64-3752-4624-99c6-2507056d1fc3.png)

### Predictions:
![image](https://user-images.githubusercontent.com/94164665/232534577-79cbc307-df16-4edf-a2de-eb101d19f224.png)

### Confusion Matrix:
![image](https://user-images.githubusercontent.com/94164665/232534721-299ee172-04ac-4a64-b0ac-6f36fcc59790.png)

### Classification Report:
![image](https://user-images.githubusercontent.com/94164665/232534892-c3390b27-f88b-4b55-992d-635768721c2c.png)


## RESULT:
Thus a Multilayer Perceptron with Backpropagation is implemented for Multi classification.



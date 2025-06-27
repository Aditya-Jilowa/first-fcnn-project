import numpy as np
import matplotlib.pyplot as plt
import random
X_train=np.loadtxt('train_X.csv', delimiter=',').T
Y_train=np.loadtxt('train_label.csv', delimiter=',').T

X_test=np.loadtxt('test_X.csv', delimiter=',').T
Y_test=np.loadtxt('test_label.csv', delimiter=',').T

print("shape of X_train :",X_train.shape)
print("shape of Y_train :",Y_train.shape)
print("shape of X_test :",X_test.shape)
print("shape of Y_test :",Y_test.shape)

index=random.randrange(0, X_train.shape[1])
plt.imshow(X_train[:,index].reshape(28,28), cmap='gray')
plt.show()

# model ->

#activation functions
def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    expX=np.exp(x)
    return expX/np.sum(expX, axis= 0)

#derivative of activation function
def derivative_tanh(x):
    return (1-np.power(x,2))

def derivative_relu(x):
    return np.array(x>0,dtype=np.float32)

#initialize parameters

def initialize_parameters(n_x,n_h,n_y):
    w1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))

    w2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    parameters = {
        "w1" : w1,
        "b1" : b1,
        "w2" : w2,
        "b2" : b2
    }

    return parameters

#forward propagation

def forward_propagation(x,parameters):
    w1=parameters['w1']
    b1=parameters['b1']
    w2=parameters['w2']
    b2=parameters['b2']

    z1 = np.dot(w1,x) + b1
    a1=relu(z1)

    z2=np.dot(w2,a1) + b2
    a2=softmax(z2)

    forward_cache = {
        "z1":z1,
        "a1":a1,
        "z2":z2,
        "a2":a2
    }

    return forward_cache

#cost function

def cost_function(a2,y):
    m=y.shape[1]
    cost=-(1/m)*np.sum(y*np.log(a2))

    return cost


#back propagation

def backward_prop(x, y, parameters, forward_cache):
    w1=parameters['w1']
    b1=parameters['b1']
    w2=parameters['w2']
    b2=parameters['b2']

    a1=forward_cache['a1']
    a2=forward_cache['a2']

    m=x.shape[1]

    dz2=(a2-y)
    dw2=(1/m)*np.dot(dz2,a1.T)
    db2=(1/m)*np.sum(dz2, axis=1, keepdims=True)

    dz1=(1/m)*np.dot(w2.T, dz2)*derivative_relu(a1)
    dw1=(1/m)*np.dot(dz1, x.T)
    db1=(1/m)*np.sum(dz1, axis=1, keepdims=True)

    gradient={
        "dw1":dw1,
        "db1":db1,
        "dw2":dw2,
        "db2":db2
    }

    return gradient



#update parameters

def update_paremeters(parameters, gradient, learning_rate):
    w1=parameters['w1']
    b1=parameters['b1']
    w2=parameters['w2']
    b2=parameters['b2']

    dw1=gradient['dw1']
    db1=gradient['db1']
    dw2=gradient['dw2']
    db2=gradient['db2']

    w1 = w1-learning_rate*dw1
    b1 = b1-learning_rate*db1
    w2 = w2-learning_rate*dw2
    b2 = b2-learning_rate*db2

    parameters = {
        "w1" : w1,
        "b1" : b1,
        "w2" : w2,
        "b2" : b2
    }

    return parameters

# complete model

def model(x, y, n_h, learning_rate, iterations):

    n_x=x.shape[0]
    n_y=y.shape[0]

    cost_list=[]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(iterations):

        forward_cache=forward_propagation(x, parameters)

        cost = cost_function(forward_cache['a2'], y)
        gradients = backward_prop(x, y, parameters, forward_cache)
        parameters = update_paremeters(parameters, gradients, learning_rate)

        cost_list.append(cost)

        if(i%(iterations/10)==0):
            print("cost after", i , "itteration is :", cost)

    return parameters, cost_list
    
iterations=100
n_h=1000
learning_rate=0.003
Parameters, Cost_list = model(X_train, Y_train, n_h=n_h, learning_rate=learning_rate, iterations=iterations)

t=np.arange(0,iterations)
plt.plot(t,Cost_list)
plt.show()


#forward_cache = forward_propagation(X_train, Parameters)
#a_out = forward_cache['a2']

#a_out=np.argmax(a_out, 0)

#print(a_out)

idx= random.randrange(0,X_test.shape[1])
plt.imshow(X_test[:,idx].reshape(28,28), cmap='gray')
plt.show()

forward_cache=forward_propagation(X_test[:, idx].reshape(X_test.shape[0], 1),Parameters)
a_out = forward_cache['a2']

a_out = np.argmax(a_out, 0)

print("out model says, it is :", a_out[0])

#save parameters
np.savez("model_parameters.npz",
         w1=Parameters["w1"],
         b1=Parameters["b1"],
         w2=Parameters["w2"],
         b2=Parameters["b2"])
print("Model parameters saved to model_parameters.npz")

import numpy as np
import random
import matplotlib.pyplot as plt

X_test=np.loadtxt('test_X.csv', delimiter=',').T

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


# Load the .npz file
data = np.load("model_parameters.npz")


# Access the saved arrays
w1 = data["w1"]
b1 = data["b1"]
w2 = data["w2"]
b2 = data["b2"]

# (Optional) Reconstruct the parameters dictionary if needed
Parameters = {
    "w1": w1,
    "b1": b1,
    "w2": w2,
    "b2": b2
}

print("Model parameters loaded successfully.")

for i in range(10):
    idx= random.randrange(0,X_test.shape[1])
    plt.imshow(X_test[:,idx].reshape(28,28), cmap='gray')
    

    forward_cache=forward_propagation(X_test[:, idx].reshape(X_test.shape[0], 1),Parameters)
    a_out = forward_cache['a2']

    a_out = np.argmax(a_out, 0)
    predicted_value=a_out[0]
    plt.title(f"Predicted value: {predicted_value}", fontsize=14)
    plt.show()

    
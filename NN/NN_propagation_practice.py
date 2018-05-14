import sys
import numpy as np
# import anything else you need

if len(sys.argv) < 3:
    print('usage: python solver.py <training_data> <testing_data>')
# you can add other arguments after <training_data> <testing_data>

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derive_sigmoid(x):
    return x * (1 - x)

# load data
a = np.loadtxt(sys.argv[1])
np.random.shuffle(a)
train_file = a[:,:2]
result = a[:,2]

b = np.loadtxt(sys.argv[2])
np.random.shuffle(b)
test_file = b[:,:2]
test_result = b[:,2]

theta_0 = 2 * np.random.rand(2, 4) - 1
theta_1 = 2 * np.random.rand(4, 2) - 1
theta_out = 2 * np.random.rand(2, 1) - 1

epoch = 1
learning_rate = 0.03

# train your model
while epoch < 1000 :

    for i in range(len(train_file)) :
    # front propagation
        layer1_input = np.dot(train_file[i], theta_0)
        layer1_activation = sigmoid(layer1_input)
        layer2_input = np.dot(layer1_activation, theta_1)
        layer2_activation = sigmoid(layer2_input)
        output_layer_input = np.dot(layer2_activation, theta_out)
        output = sigmoid(output_layer_input)

    # back propagation
        E = result[i] - output
        slope_output = derive_sigmoid(output)
        slope_layer2 = derive_sigmoid(layer2_activation)
        slope_layer1 = derive_sigmoid(layer1_activation)
        d_output = E * slope_output
        d_layer2 = d_output.dot(theta_out.T) * slope_layer2
        d_layer1 = d_layer2.dot(theta_1.T) * slope_layer1
        theta_out += np.outer(layer2_activation, d_output) * learning_rate
        theta_1 += np.outer(layer1_activation, d_layer2) * learning_rate
        theta_0 += np.outer(train_file[i], d_layer1) * learning_rate

    correct = 0
    for j in range(len(test_file)) :
        # front propagation
        layer1_input = np.dot(test_file[j], theta_0)
        layer1_activation = sigmoid(layer1_input)
        layer2_input = np.dot(layer1_activation, theta_1)
        layer2_activation = sigmoid(layer2_input)
        output_layer_input = np.dot(layer2_activation, theta_out)
        output = sigmoid(output_layer_input)
        if (output >= 0.5 and test_result[j] == 1) or (output < 0.5 and test_result[j] == 0) :
            correct += 1

    epoch += 1
    # print train loss and test accuracy after every epoch
    print("Epoch = ", epoch)
    print("Train loss = ", np.mean(np.square(E)))
    print("Test accuracy = ", correct/len(test_file))


# save weight


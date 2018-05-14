import numpy as np

max_iterations = 1500
alpha = 0.01

def gen_gradient(x, y, max_iterations, alpha):
    theta0 = 0
    theta1 = 0
    iter = 0
    converged = False

    while not converged:
        theta_0 = theta0 - alpha * (1/m * sum([(theta0 + theta1 * x[i] - y[i]) for i in range(m)]))
        theta_1 = theta1 - alpha * (1/m * sum([(theta0 + theta1 * x[i] - y[i]) * x[i] for i in range(m)]))
        temp0 = theta_0
        temp1 = theta_1
        theta0 = temp0
        theta1 = temp1
        print (theta0)
        print (theta1)
        iter += 1
        if iter == max_iterations:
            converged = True

    return theta0, theta1

f = open('ex1data1.txt', 'r')
x = []
y = []
while True :
    i = f.readline()
    if i == "" : break
    i = i.rstrip()
    i = i.split(',', 1)
    x.append(float(i[0]))
    y.append(float(i[1]))

m = len(x)

theta0, theta1 = gen_gradient(x, y, max_iterations, alpha)

predit = theta0 + theta1 * 3.5
print ('predit 3.5 = ' , predit)

predit = theta0 + theta1 * 7
print ('predit 7 = ' , predit)
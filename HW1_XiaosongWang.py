# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:51:27 2020

@author: simon
"""

# 1 Import packages
import numpy as np
import matplotlib.pyplot as plt

# 2 Creating training dataset
x_data = np.array([13., 7., 4., 20., 22., 25., 17., 60., 8., 60.])
y_data = 2 * x_data + 50 + 5 * np.random.random()

# 3 Get the loss function
bias =  np.arange(0, 100, 1)
weight = np.arange(-5, 5, 0.1)
Z = np.zeros((len(bias),len(weight)))

for i in range(len(bias)):
    for j in range(len(weight)):
        b = bias[i]
        w = weight[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (w * x_data[n] + b - y_data[n]) ** 2

# 4 Initialize w and b
w = 0
b = 0

# 5 Set the learning rate, maximum iteration, and a iteration counter
lr = 0.0001
iteration_max = 50000
iteration_curr = 0

# 6 Create lists for w and b
b_history = [b]
w_history = [w]

# Find w and b by gradient descent and checking convergence for terminating the loop
for i in range(iteration_max):
    w_d = 0.0
    b_d = 0.0
    
    lost_pre = 0.5 * sum(((b + w * x_data[j] - y_data[j]) ** 2 for j in range(len(x_data))))      
      
    for n in range(len(x_data)):
        y_pred = w * x_data[n] + b
        w_d = w_d + ((y_pred - y_data[n]) * x_data[n])
        b_d = b_d + (y_pred - y_data[n])
        w = w - lr * w_d
        b = b - lr * b_d
    
    lost_curr = 0.5 * sum(((b + w * x_data[j] - y_data[j]) ** 2 for j in range(len(x_data))))

    b_history.append(b)
    w_history.append(w)
    
    iteration_curr += 1
    
    #Checking convergence
    if abs(lost_pre - lost_curr) <= 0.00000000000000000000000000000001 :
        break
            
print('w:', w, ' b:', b, 'iteration:', iteration_curr)
    
# Plot the graph
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5,color='black')
plt.plot(b, w, 'x', ms = 10, color = 'orange')
plt.xlim(0,100)
plt.ylim(-5,5)
plt.contourf(bias,weight,Z,50, alpha = 0.5, cmap = plt.get_cmap('jet'))
plt.show 

# testing model
x_testing = np.array([5., 17., 13., 21., 23., 27., 31., 26., 4., 51.])
y_testing = 2 * x_data + 50 + 5 * np.random.random()


for i in range(len(x_testing)):
    error = 0.
    error = error + (abs(w * x_testing[i] + b - y_testing[i])) / y_testing[i]

error_avg = error / len(x_testing) * 100

print('The error is: ', error_avg, '%')
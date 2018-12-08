import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # To ignore AVX2 FMA extensions

data=pd.read_csv('FuelConsumption.csv')

train_x = np.asanyarray(data[['ENGINESIZE','CYLINDERS','FUELTYPE']])
train_y = np.asanyarray(data[['CO2EMISSIONS']])

#Fueltype is of Stringtype. Hence converting it to equivalent integer
for i in range(train_x.shape[0]):
    train_x[i][2] = ord(train_x[i][2][0])-ord('A')


#slicing to separate test data and training data
test_x = train_x[801:]
test_x = np.c_[np.ones(test_x.shape[0]),test_x]
test_y = train_y[801:]
test_y_forplotting = test_y

train_x = train_x[:801]
train_x = np.c_[np.ones(train_x.shape[0]),train_x]
train_y = train_y[:801]

# Define all the tensors and nodes of the graph

test_x = tf.convert_to_tensor(test_x,dtype=tf.float32)
test_y = tf.convert_to_tensor(test_y,dtype=tf.float32)
train_x = tf.convert_to_tensor(train_x,dtype=tf.float32)
train_y = tf.convert_to_tensor(train_y,dtype=tf.float32)
theta = tf.Variable(tf.zeros([train_x.shape[1],1],dtype=tf.float32))
output = tf.matmul(train_x,theta)
test_output = tf.round(tf.matmul(test_x,theta))
cost_function = tf.reduce_mean(tf.square(output-train_y))
optimizer = tf.train.GradientDescentOptimizer(0.0015)
train = optimizer.minimize(cost_function)
init = tf.global_variables_initializer()

# Run the session
J_theta = []
with tf.Session() as sess:
    sess.run([init,train_x,train_y,test_x,test_y])
    for step in range(10000):
        values = sess.run([theta,cost_function,train])
        J_theta.append(values[1])
    Predicted_output=sess.run(test_output)

# Plot results
plt.plot(J_theta[:50])
plt.ylabel('Cost_function')
plt.xlabel('Iteration')
plt.show()

fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(Predicted_output, 'C1', label='Predicted_output')
ax.plot(test_y_forplotting, 'C2', label='Dataset')
ax.legend()
plt.show()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # To ignore AVX2 FMA extensions
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
input->weight->hiddenlayer1->activation fn->weight->hiddenlayer2->activation fn->weights->output

Above process is called feed forward

Then cost function is caluclated and optimized by backpropagation.

feedforward + backpropagation = epoch (1 cycle)
'''

# Manipulation of dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

y_train_exp = np.array([[0]*10]*len(y_train))
y_test_exp = np.array([[0]*10]*len(y_test))
for i in range(len(y_train)):
    y_train_exp[i][y_train[i]] = 1

for i in range(len(y_test)):
    y_test_exp[i][y_test[i]] = 1

x_train = x_train.reshape([x_train.shape[0],x_train.shape[1]*x_train.shape[2]])
x_test = x_test.reshape([x_test.shape[0],x_test.shape[1]*x_test.shape[2]])

# ---------------------------------------------------------------------------------

hl1_size = 500
hl2_size = 500
hl3_size = 500
output_layer_size = 10
batch_size = 10
no_of_train_examples = x_train.shape[0]
cost_values=[]

x = tf.placeholder(tf.float32,[None,x_train.shape[1]])
y = tf.placeholder(tf.float32,[None,y_train_exp.shape[1]])

def neural_network(data):

    hidden_layer_1 = { 'weights': tf.Variable(tf.random_normal([x_train.shape[1],hl1_size])),
                       'biases': tf.Variable(tf.random_normal([hl1_size])) }

    hidden_layer_2 = { 'weights': tf.Variable(tf.random_normal([hl1_size,hl2_size])),
                       'biases': tf.Variable(tf.random_normal([hl2_size])) }

    hidden_layer_3 = { 'weights': tf.Variable(tf.random_normal([hl2_size,hl3_size])),
                       'biases': tf.Variable(tf.random_normal([hl3_size])) }

    output_layer = { 'weights': tf.Variable(tf.random_normal([hl3_size,output_layer_size])),
                       'biases': tf.Variable(tf.random_normal([output_layer_size])) }

    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1) #like a activation function

    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])

    return output

def train_neural_network(data):

    prediction = neural_network(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels= y,logits=prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    epochs = 10
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            loss = 0
            for j in range(int(no_of_train_examples/batch_size)):
                x_batch_data = x_train[j*batch_size:j*batch_size+batch_size,:]
                y_batch_data = y_train_exp[j*batch_size:j*batch_size+batch_size,:]
                _,c = sess.run([optimizer,cost],feed_dict={x:x_batch_data,y:y_batch_data})
                loss += c

            cost_values.append(loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test_exp}))

train_neural_network(x)
print(cost_values)
plt.style.use('ggplot')
plt.plot(cost_values)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()

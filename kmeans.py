import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

number_of_points = 200
K = 7
iterations = 100

points_init = tf.constant(np.random.uniform(0, 10, (number_of_points, 2)))
centroids_init = tf.Variable(tf.slice(tf.random_shuffle(points_init), [0, 0], [K, -1])) #tf.slice(input,start,size)
init = tf.global_variables_initializer()

"""
    The dimesions are expanded so as to find the difference of
    each train data with every centroid without using loops.

    A really smart way I found online.
"""

points = tf.expand_dims(points_init, 0)
centroids = tf.expand_dims(centroids_init, 1)

distance_with_centroids = tf.reduce_sum(tf.square(tf.subtract(points, centroids)), 2)
closest_centroid = tf.argmin(distance_with_centroids, 0)


means = []
for i in range(K):
    means.append(tf.reduce_mean(tf.gather(points_init,tf.reshape(tf.where(tf.equal(closest_centroid, i)),[-1]),1),0))

new_centroids = tf.concat([means],0)
update_centroids = tf.assign(centroids_init, new_centroids)

with tf.Session() as sess:
    sess.run(init)
    for i in range(iterations):
        [_, centroid_val, points_val, close] = sess.run([update_centroids, centroids_init, points_init, closest_centroid])
    print ("Centroids: ", centroid_val)

plt.scatter(points_val[:, 0], points_val[:, 1], c=close, s=50, alpha=0.5)
plt.plot(centroid_val[:, 0], centroid_val[:, 1], 'kx', markersize=15)
plt.show()

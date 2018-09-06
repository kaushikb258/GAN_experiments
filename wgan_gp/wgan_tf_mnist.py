import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 784])

D_W1 = tf.Variable(xavier_init([784, 512]))
D_b1 = tf.Variable(tf.zeros(shape=[512]))

D_W2 = tf.Variable(xavier_init([512, 256]))
D_b2 = tf.Variable(tf.zeros(shape=[256]))

D_W3 = tf.Variable(xavier_init([256, 1]))
D_b3 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(xavier_init([100, 256]))
G_b1 = tf.Variable(tf.zeros(shape=[256]))

G_W2 = tf.Variable(xavier_init([256, 512]))
G_b2 = tf.Variable(tf.zeros(shape=[512]))

G_W3 = tf.Variable(xavier_init([512, 784]))
G_b3 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]


def sample_Z(m, n):
    return np.random.uniform(low=-1., high=1., size=[m, n]).astype(np.float32)


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_log_prob = tf.matmul(G_h2, G_W3) + G_b3
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    out = tf.matmul(D_h2, D_W3) + D_b3
    return out



def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


#G_sample = generator(Z)
#D_real, D_logit_real = discriminator(X)
#D_fake, D_logit_fake = discriminator(G_sample)

#D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
#G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
#D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
#D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
#D_loss = D_loss_real + D_loss_fake
#G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

#D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
#G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


#--------------------------------------------------------------------

X_fake = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(X_fake)

g_loss = -tf.reduce_mean(D_fake)
d_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)

epsilon = tf.random_uniform([], 0.0, 1.0)
x_hat = epsilon * X + (1 - epsilon) * X_fake
d_hat = discriminator(x_hat)

ddx = tf.gradients(d_hat, x_hat)[0]
ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
ddx_2 = tf.reduce_mean(tf.square(ddx - 1.0))*10.0

d_loss = d_loss + ddx_2

d_adam = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=theta_D)
g_adam = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=theta_G)



#--------------------------------------------------------------------
saver = tf.train.Saver()

irestart = 0

sess = tf.Session()


if (irestart == 0):
      sess.run(tf.global_variables_initializer())
else:
      saver.restore(sess, "ckpt/model")  

#--------------------------------------------------------------------

mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    if it % 25000 == 0:
        samples = sess.run(X_fake, feed_dict={Z: sample_Z(64, Z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)

   
    Zn = sample_Z(mb_size, Z_dim)
    Zn = np.array(Zn).astype(np.float32)
    _, D_loss_curr = sess.run([d_adam, d_loss], feed_dict={X: X_mb, Z: Zn})
     
    _, G_loss_curr = sess.run([g_adam, g_loss], feed_dict={Z: Zn})

     
    if it % 100 == 0:
      Dr = sess.run([tf.reduce_mean(D_real)], feed_dict={X: X_mb, Z: Zn})
      Df = sess.run([tf.reduce_mean(D_fake)], feed_dict={X: X_mb, Z: Zn})
      Dr = np.array(Dr)[0]
      Df = np.array(Df)[0]
      #print("Dr: ", np.round(Dr,3), "Df: ", np.round(Df,3), "Wasserstein : ", np.round(Dr - Df,3))
 
      E_ddx = sess.run([tf.reduce_mean(ddx)], feed_dict={X: X_mb, Z: Zn})
      E_ddx = np.array(E_ddx)[0]
      #print("E_ddx: ", E_ddx)

      with open("wasserstein.txt", "a") as myfile:
        myfile.write(str(it) + " " + str(Dr) + " " + str(Df) + " " + str(Dr - Df) + " " + str(E_ddx) + " " + str(D_loss_curr) + " " + str(G_loss_curr) + "\n")  



    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()



saver.save(sess, "ckpt/model")


   

from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog
import sys
import mnist_dialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class mnistApp(QtWidgets.QDialog , mnist_dialog.Ui_Dialog):
    def __init__(self, parant=None):
        super(mnistApp, self).__init__(parant)
        self.setupUi(self)
        
        # TODO : ser hyperparameters
        self.batch_size = 32
        self.learning_rate = 0.05
        self.optimizer = 'SGD'
        self.num_steps = 700

        self.btn1.clicked.connect(self.btn1_Clicked)
        self.btn2.clicked.connect(self.btn2_Clicked)
        self.btn3.clicked.connect(self.btn3_Clicked)
        self.btn4.clicked.connect(self.btn4_Clicked)
        self.btn5.clicked.connect(self.btn5_Clicked)

    def btn1_Clicked(self):
        trained_image = mnist.train.images
        trained_label = mnist.train.labels
        rand_num = 10
        rand_arr = np.random.randint(trained_image.shape[0], size = rand_num)

        for value in rand_arr:
            curr_image = np.reshape(trained_image[value, :], (28,28))
            curr_label = np.argmax(trained_label[value, :])
            plt.matshow(curr_image, cmap = 'gray')
            plt.title("Label: "+ str(curr_label))
            plt.show()

    def btn2_Clicked(self):
        print("hyperparamters:")
        print("batch size: %d" % self.batch_size)
        print("learning rate: %f" % self.learning_rate)
        print("optimizer: %s" % self.optimizer)

    def btn3_Clicked(self):

        graph = tf.Graph()
        with graph.as_default():

            tf_train_dataset = tf1.placeholder(tf.float32 ,[None,784])
            tf_train_labels = tf1.placeholder(tf.float32, [None, 10])

            logits = LeNet(tf_train_dataset)
            # loss
            loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits))

            ## optimizer
            optimizer = tf1.train.GradientDescentOptimizer(self.learning_rate).minimize(loss_func)

            with tf1.Session(graph = graph) as session:
                plt.title("epoch [0/50]")
                plt.xlabel("iteration")
                plt.ylabel("loss")
                x = []
                y = []
                tf1.global_variables_initializer().run()
                for step in range(600):
                    batch_data, batch_labels = mnist.train.next_batch(self.batch_size)
                    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                    loss = session.run([optimizer, loss_func], feed_dict = feed_dict)
                    if step%2==0:
                        x.append(step)
                        y.append(loss)
               
                plt.plot(x,y,'-')
                plt.savefig("Q5_3")
                plt.show()

    def btn4_Clicked(self):
        # self.CNN_model()
        image = mpimg.imread('Q5_4.png')
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.show()

    def btn5_Clicked(self):
        graph = tf.Graph()
        with graph.as_default():

            tf_train_dataset = tf1.placeholder(tf.float32, [None,784])
            tf_train_labels = tf1.placeholder(tf.float32, [None, 10])
            
            train_prediciton = LeNet(tf_train_dataset)
            
            ## loss
            loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = train_prediciton))

            ## optimizer
            optimizer = tf1.train.GradientDescentOptimizer(self.learning_rate).minimize(loss_func)

            ## accuracy
            correct_prediction = tf.equal(tf.argmax(train_prediciton,1),tf.argmax(tf_train_labels,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            
            saver = tf1.train.Saver()
            with tf1.Session(graph = graph) as session:        
                save_path = "./net/save_net.ckpt"
                saver.restore(session, save_path)
                index = int(self.lineEdit.text())
                pic = np.array([mnist.test.images[index]])
                pre_ans = session.run(train_prediciton, feed_dict = {tf_train_dataset:pic})
                curr_image = np.reshape(mnist.test.images[index, :], (28,28))
                plt.matshow(curr_image, cmap = 'gray')
                

                X = [0,1,2,3,4,5,6,7,8,9]
                plt.figure()
                plt.xticks(np.linspace(0,9,10))
                plt.bar(X,list(pre_ans[0]),width=0.5)
                plt.show()
                


    def CNN_model(self):

        graph = tf.Graph()
        with graph.as_default():

            tf_train_dataset = tf1.placeholder(tf.float32, [None,784])
            tf_train_labels = tf1.placeholder(tf.float32, [None, 10])
            
            train_prediciton = LeNet(tf_train_dataset)
            
            ## loss
            loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = train_prediciton))

            ## optimizer
            optimizer = tf1.train.GradientDescentOptimizer(self.learning_rate).minimize(loss_func)

            ## accuracy
            correct_prediction = tf.equal(tf.argmax(train_prediciton,1),tf.argmax(tf_train_labels,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            
            with tf1.Session(graph = graph) as session:
                x = []
                y_train_acc = []
                y_valid_acc = []
                y_train_loss = []
                y_valid_loss = []

                tf1.global_variables_initializer().run()
                for epoch in range(50):
                    for step in range(self.num_steps):
                        batch_data, batch_labels = mnist.train.next_batch(self.batch_size)
                        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                        session.run(optimizer, feed_dict = feed_dict)

                    loss_t,acc_t = session.run([loss_func,accuracy],feed_dict = {tf_train_dataset : mnist.train.images, tf_train_labels : mnist.train.labels})
                    loss,acc = session.run([loss_func,accuracy],feed_dict = {tf_train_dataset : mnist.validation.images, tf_train_labels : mnist.validation.labels})
                    print("Epoch: %02d loss_t: %.9f Acc_t:%.4f" %(epoch+1 , loss_t, acc_t))
                    print("Epoch: %02d loss: %.9f Acc:%.4f" %(epoch+1 , loss, acc))
                    x.append(epoch+1)
                    y_train_acc.append(acc_t*100.0)
                    y_valid_acc.append(acc*100.0)
                    y_train_loss.append(loss_t)
                    y_valid_loss.append(loss)

                fig, axs = plt.subplots(2,1,constrained_layout = True)
                t1 = axs[0].plot(x,y_train_acc,'-')
                t2 = axs[0].plot(x,y_valid_acc,'-')
                axs[0].set_title("Accuracy")
                axs[0].set_xlabel("epoch")
                axs[0].set_ylabel("%")
                axs[0].legend([t1,t2], labels = ['training' , 'testing'], loc = 'lower right')

                axs[1].set_title("Loss")
                axs[1].set_xlabel("epoch")
                axs[1].set_ylabel("loss")
                t3 = axs[1].plot(x,y_train_loss,'-')
                t4 = axs[1].plot(x,y_valid_loss,'-')
                axs[1].legend([t3,t4], labels = ['training' , 'testing'], loc = 'upper right')
                
                fig.savefig("Q5_4")

                # Save model
                saver = tf1.train.Saver()
                saver.save(session, "net/save_net.ckpt")

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def LeNet(X):

    X_ = tf.reshape(X, [-1, 28, 28, 1])
    ## conv1
    W_conv1 = weight_variable(shape = [5,5,1,6])
    b_conv1 = bias_variable(shape = [6])
    A_conv1 = tf.nn.relu(tf.nn.conv2d(X_, W_conv1, strides = [1,1,1,1],padding = 'SAME') + b_conv1)

    ## maxpool1
    A_pool1 = tf.nn.max_pool(A_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    ## conv2
    W_conv2 = weight_variable(shape = [5,5,6,16])
    b_conv2 = bias_variable(shape = [16])
    A_conv2 = tf.nn.relu(tf.nn.conv2d(A_pool1, W_conv2,strides = [1,1,1,1],padding = 'VALID') + b_conv2)

    ## maxpool2
    A_pool2 = tf.nn.max_pool(A_conv2,ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'SAME')

    ## FC3
    A_pool2_flat = tf.reshape(A_pool2, [-1 ,5*5*16])

    W_fc3 = weight_variable([5*5*16, 120])
    b_fc3 = bias_variable([120])
    A_fc3 = tf.nn.relu(tf.matmul(A_pool2_flat, W_fc3)+b_fc3)

    ## FC4
    W_fc4 = weight_variable([120,84])
    b_fc4 = bias_variable([84])
    A_fc4 = tf.nn.relu(tf.matmul(A_fc3,W_fc4) + b_fc4)

    ## Softmax
    W_1 = weight_variable([84,10])
    b_1 = bias_variable([10])
    A_1 = tf.nn.softmax(tf.matmul(A_fc4,W_1)+b_1)

    return A_1

def main():
    app = QApplication(sys.argv)
    form = mnistApp()
    form.show()
    app.exec_()

if __name__ == '__main__' :
    main()
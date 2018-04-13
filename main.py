import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
from skimage import io
import time
import random
import matplotlib.pyplot as plt
from layers import *
from model import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

to_train = True
to_test = False
to_restore = True
filenames_A,filenames_B="./input/horse2zebra/trainA/",  "./input/horse2zebra/trainB/"
out_file1,out_file2='./A2B_fake_B/','./B2A_fake_A/'
output_path = "./output"
check_dir = "./output/checkpoints/"
temp_check = 0

max_epoch = 80
max_images = 1000

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
pool_size = 50
sample_size = 10
save_training_images = True
ngf = 32
ndf = 64
A_imgid=0
B_imgid=0
class CycleGAN():
    def Img_to_RGB(self,img):
        eps=1e-8
        return img/(127.5+eps)-1
    def RGB_to_Img(self,img):
        return ((img+1)*127.5).astype(np.uint8)
    def get_batch(self, ptr, batch_size=batch_size):
        start,end=ptr*batch_size,min(ptr*batch_size+batch_size,max_images)
        return self.A_input[start:end,:],self.B_input[start:end,:]
    def load_data(self,main_path):
        filenames=os.listdir(main_path)
        filenames = sorted(filenames, key=lambda x: int(x[x.find('_') + 1:x.find('.')]))
        res=np.zeros([max_images, batch_size, img_width, img_height, img_layer])
        n_256256=0
        num=0
        for filename in filenames:
            now_img=io.imread(main_path+filename)#.astype(np.float32)
            now_img=self.Img_to_RGB(now_img)
            shape=now_img.shape
            if len(shape)<3:
                n_256256+=1
                # print(main_path+"- file name: ",filename)
                continue
            res[num]=now_img
            num+=1
            if num==max_images:
                break
        # print('256*256 number: ', n_256256)
        # print('256*256*3 number: ', len(filenames)-n_256256)
        res=np.array(res)
        return res,len(filenames)
    def input_setup(self):
        self.A_input, self.queue_length_A = self.load_data(filenames_A)
        self.B_input, self.queue_length_B = self.load_data(filenames_B)
        self.fake_images_A = np.zeros((pool_size, 1, img_height, img_width, img_layer))
        self.fake_images_B = np.zeros((pool_size, 1, img_height, img_width, img_layer))
        self.test_input_A, self.test_input_B = self.get_batch(0, batch_size=16)
        # print('**********************************feed_input_A****************************')
        plt.imsave(out_file1 + '{}.png'.format(str(0).zfill(3)), self.RGB_to_Img(self.test_input_A[A_imgid,0,:]))
        plt.imsave(out_file2 + '{}.png'.format(str(0).zfill(3)), self.RGB_to_Img(self.test_input_B[B_imgid,0,:]))

    def model_setup(self):
        self.input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")
        self.fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_B")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.num_fake_inputs = 0
        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")
        with tf.variable_scope("Model") as scope:
            self.fake_B = build_generator_resnet_9blocks(self.input_A, name="g_A")
            self.fake_A = build_generator_resnet_9blocks(self.input_B, name="g_B")
            self.rec_A = build_gen_discriminator(self.input_A, "d_A")
            self.rec_B = build_gen_discriminator(self.input_B, "d_B")
            scope.reuse_variables()
            self.fake_rec_A = build_gen_discriminator(self.fake_A, "d_A")
            self.fake_rec_B = build_gen_discriminator(self.fake_B, "d_B")
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A, "g_A")
            scope.reuse_variables()
            self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A, "d_A")
            self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B, "d_B")

    def loss_calc(self):
        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B,1))
        self.g_loss_A = cyc_loss*10 + disc_loss_B
        self.g_loss_B = cyc_loss*10 + disc_loss_A
        self.d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(self.rec_A,1)))/2.0
        self.d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(self.rec_B,1)))/2.0
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        self.model_vars = tf.trainable_variables()
        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]
        self.d_A_trainer = optimizer.minimize(self.d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(self.d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(self.g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(self.g_loss_B, var_list=g_B_vars)
        # for var in self.model_vars: print(var.name)

    def save_training_images(self, sess, epoch):

        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")

        for i in range(0,10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
            imsave("./output/imgs/fakeB_"+ str(epoch) + "_" + str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/fakeA_"+ str(epoch) + "_" + str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/cycA_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_A_temp[0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/cycB_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_B_temp[0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/inputA_"+ str(epoch) + "_" + str(i)+".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/inputB_"+ str(epoch) + "_" + str(i)+".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))
    def show_one(self,sess,epoch,num):
        batch_input_A, batch_input_B = self.test_input_A,self.test_input_B
        fake_B_temp = sess.run(self.fake_B,feed_dict={self.input_A: batch_input_A[0]})
        fake_A_temp = sess.run(self.fake_A,feed_dict={self.input_B: batch_input_B[0]})
        ###########################################################
        name_B=out_file1+str(epoch)+'_'+str(num + 1).zfill(3)+'.png'
        name_A=out_file2+str(epoch)+'_'+str(num + 1).zfill(3)+'.png'
        plt.imsave(name_B,self.RGB_to_Img(fake_B_temp[A_imgid,:]))
        plt.imsave(name_A,self.RGB_to_Img(fake_A_temp[B_imgid,:]))
        return num + 1

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        ''' This function saves the generated image to corresponding pool of images.
        In starting. It keeps on feeling the pool till it is full and then randomly selects an
        already stored image and replace it with new one.'''
        if(num_fakes < pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else :
                return fake

    def train(self):
        print('\nStarting data processing:\n')
        self.input_setup()
        print('\nCreating model:\n')
        self.model_setup()
        print('\nCreating loss function:\n')
        self.loss_calc()
        print('\nStart training:\n')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            # Restore the model to run the model from last checkpoint
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)
            if not os.path.exists(check_dir):
                os.makedirs(check_dir)
            start_time = time.time()
            step = 100
            D_A_error, G_A_error, D_B_error, G_B_error = [], [],[],[]
            for epoch in range(sess.run(self.global_step),max_epoch):
                print ("In the epoch ", epoch)
                saver.save(sess, os.path.join(check_dir, "cyclegan"), global_step=epoch)
                # Dealing with the learning rate as per the epoch number
                if(epoch < 100) :
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100

                if(save_training_images):
                    self.save_training_images(sess, epoch)
                num=0
                for ptr in range(0,max_images):
                    print('In the iteration ', ptr, '---- Time used %.1f s' % (time.time() - start_time), end='\r')
                    if ptr % step == 0:
                        # num = self.show(sess,num)
                        num = self.show_one(sess,epoch,num)
                    # Optimizing the G_A network

                    _, fake_B_temp, g_loss_A = sess.run([self.g_A_trainer, self.fake_B, self.g_loss_A],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})

                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)
                    
                    # Optimizing the D_B network
                    _, d_loss_B = sess.run([self.d_B_trainer, self.d_loss_B],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_B:fake_B_temp1})
                    
                    # Optimizing the G_B network
                    _, fake_A_temp, g_loss_B = sess.run([self.g_B_trainer, self.fake_A, self.g_loss_B],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})

                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, d_loss_A = sess.run([self.d_A_trainer, self.d_loss_A],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_A:fake_A_temp1})

                    self.num_fake_inputs+=1

                    G_A_error.append(g_loss_A)
                    D_B_error.append(d_loss_B)
                    G_B_error.append(g_loss_B)
                    D_A_error.append(d_loss_A)
                sess.run(tf.assign(self.global_step, epoch + 1))
        x = range(len(D_A_error))
        plt.plot(x, D_A_error,'-r', x, D_B_error,'-g',
                 x, G_A_error, '-b',x, G_B_error,'-m')
        plt.legend(['Discriminator A', 'Discriminator B', 'Generator A', 'Generator B'])
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
        print('mean errror for D_A, D_B, G_A, G_B: ',
              np.mean(D_A_error), np.mean(D_B_error), np.mean(G_A_error), np.mean(G_B_error))
    def test(self):

        ''' Testing Function'''

        print("Testing the results")

        self.input_setup()

        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            self.input_read(sess)

            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)

            if not os.path.exists("./output/imgs/test/"):
                os.makedirs("./output/imgs/test/")            

            for i in range(0,100):
                fake_A_temp, fake_B_temp = sess.run([self.fake_A, self.fake_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
                imsave("./output/imgs/test/fakeB_"+str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/fakeA_"+str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/inputA_"+str(i)+".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/inputB_"+str(i)+".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))

def main():
    
    model = CycleGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()

if __name__ == '__main__':

    main()
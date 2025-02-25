#!/usr/bin/env python
# coding: utf-8

# # Pix2Pix 
# 
# Following code Separately store the weights each layer of generator network

# ## Import TensorFlow and other libraries

from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import time

from matplotlib import pyplot as plt
from IPython import display
import numpy as np
from termcolor import colored

print('tensorflow version :: ',tf.__version__)


PATH = 'dataset/'
BUFFER_SIZE = 10
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, w:, :]
    input_image = image[:, :w, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return real_image,input_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_batchnorm=True, name='downsample'):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',name=name+'_conv',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False, name='upsample'):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    name=name+'_transposeconv',
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False,name='downsample_1'), # (bs, 128, 128, 64)
    downsample(128, 4,name='downsample_2'), # (bs, 64, 64, 128)
    downsample(256, 4,name='downsample_3'), # (bs, 32, 32, 256)
    downsample(512, 4,name='downsample_4'), # (bs, 16, 16, 512)
    downsample(512, 4,name='downsample_5'), # (bs, 8, 8, 512)
    downsample(512, 4,name='downsample_6'), # (bs, 4, 4, 512)
    downsample(512, 4,name='downsample_7'), # (bs, 2, 2, 512)
    downsample(512, 4,name='downsample_8'), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True,name='upsample_1'), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True,name='upsample_2'), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True,name='upsample_3'), # (bs, 8, 8, 1024)
    upsample(512, 4,name='upsample_4'), # (bs, 16, 16, 1024)
    upsample(256, 4,name='upsample_5'), # (bs, 32, 32, 512)
    upsample(128, 4,name='upsample_6'), # (bs, 64, 64, 256)
    upsample(64, 4,name='upsample_7'), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh',
                                         name='upsample_8') # (bs, 256, 256, 3)

  x = inputs
  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    print(colored(x.name + ' shape :: {} '.format(x.shape),'yellow', attrs=['blink']))
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    print(colored(x.name + ' shape :: {} '.format(x.shape),'cyan', attrs=['blink']))
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  print(colored(x.name + ' shape :: {} '.format(x.shape),'cyan', attrs=['blink']))

  return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()
#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

print('='*50)
text = colored('Total Trainable parameters of Generator are :: {}'.format(generator.count_params()), 'red', attrs=['reverse','blink'])
print(text)
print('='*50)

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
    
    print(colored(x.name + ' shape :: {} '.format(x.shape),'magenta', attrs=['blink']))

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    print(colored(down1.name + ' shape :: {} '.format(down1.shape),'magenta', attrs=['blink']))
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    print(colored(down2.name + ' shape :: {} '.format(down2.shape),'magenta', attrs=['blink']))
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
    print(colored(down3.name + ' shape :: {} '.format(down3.shape),'magenta', attrs=['blink']))

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
    print(colored(conv.name + ' shape :: {} '.format(conv.shape),'magenta', attrs=['blink']))

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
    print(colored(last.name + ' shape :: {} '.format(last.shape),'magenta', attrs=['blink']))

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
#tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

print('='*50)
text = colored('Total Trainable parameters of Discriminator are :: {}'.format(discriminator.count_params()), 'blue', attrs=['reverse','blink'])
print(text)
print('='*50)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
try:
  os.mkdir('Training_Samples')
except:
  pass
def generate_images(model, test_input, tar, number, folder = 'Training_Samples', mode='train'):
    with tf.device('/device:cpu:0'):
        if mode == 'train' :
            prediction = model(test_input, training=True)
            display_list = [test_input[0], tar[0], prediction[0]]
            title = ['Input', 'Target', 'Output']
            try :
                os.mkdir(folder+'/Train_Validate')
            except:
                pass    
            for i in range(3):
                plt.imsave(folder+'/Train_Validate/{}_{}.png'.format(number,title[i]), np.array((display_list[i] * 0.5 + 0.5)*255, dtype='uint8'))
        elif mode == 'test' :
            prediction = model(test_input, training=True)
            display_list = [test_input[0], tar[0], prediction[0]]
            title = ['Input', 'Target', 'Output']
            try :
                os.mkdir(folder)
            except:
                pass    

            for i in range(3):
                plt.imsave(folder+'/{}_{}.png'.format(number,title[i]), np.array((display_list[i] * 0.5 + 0.5)*255, dtype='uint8'))
        else:
            print('Enter valid mode eighter [!]train or [!]test')
            exit(0)


EPOCHS = 5

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target, epoch, folder='Training_Samples',mode='train')
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)


# Now run the training loop:

fit(train_dataset, EPOCHS, test_dataset)


# ## Restore the latest checkpoint and test
# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# ## Generate using test dataset

# Run the trained model on a few examples from the test dataset
flag=1
try:
  os.mkdire('Test_Data')
except:
  pass
for inp, tar in test_dataset.take(5):
    generate_images(generator, inp, tar, flag,folder='Test_Data',mode='test')


# In[129]:


try:
    os.mkdir('Weights')
except:
    pass

model_layers = []

for layer in generator.layers:
    if 'sequential' in layer.name:
        for layer2 in layer.layers:
            if '_conv' in layer2.name:
                model_layers.append(layer2)
            elif '_transposeconv' in layer2.name:
                model_layers.append(layer2)
    elif 'upsample_8' in layer.name:
                model_layers.append(layer)
    
names = [
        'downsample_1',
        'downsample_2',
        'downsample_3',
        'downsample_4',
        'downsample_5',
        'downsample_6',
        'downsample_7',
        'downsample_8',
        'upsample_1',
        'upsample_2',
        'upsample_3',
        'upsample_4',
        'upsample_5',
        'upsample_6',
        'upsample_7',
        'upsample_8',
        ]
for name in names :
    try:
        folder = 'Weights/'+ name
        os.mkdir(folder)
    except:
        pass
    tmp1,tmp2,tmp3 = 1,1,1
    for layer in model_layers:
        if name in layer.name:
            try:
                np.save(folder+'/{}.npy'.format(layer.name),np.array(layer.get_weights(),dtype=np.float32))
            except:
                np.save(folder+'/{}_weights.npy'.format(layer.name),np.array(layer.get_weights()[0],dtype=np.float32))
                np.save(folder+'/{}_bias.npy'.format(layer.name),np.array(layer.get_weights()[1],dtype=np.float32))
                

print(colored('Saperately weights of each convolution saved successfully !!!! ','magenta', attrs=['blink']))
print('='*100)




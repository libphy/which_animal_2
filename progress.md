# Exercise with tensorflow VGG-Net
## Goal
The goal of this mini exercise is to learn how to use trained VGGNet in tensorflow.
See [Karpathy's note about transfer learning](http://cs231n.github.io/transfer-learning/) for how to use trained net.
I have an extension idea on my ['which_animal' project](https://github.com/libphy/which_animal) to recognizing animals from images and somehow combine with sound information. To begin with, I decided to use cats and dogs images from [kaggle](https://www.kaggle.com/c/dogs-vs-cats) as my new dataset, then use a VGG-Net pre-trained using ImageNet data which has its own cats and dogs images- so the new data from kaggle would be smaller subset of categories in ImageNet however the images are not identical but similar.  

## Tools
I have struggled to make Keras+Tensorflow to work on the Keras version of pretrained VGGNet. It turned out that Keras is stable on theano backend, but has many issues in tensorflow backend in terms of memory management. There was no problem when I used small size data and small size model (I used 6 convolutional layers for sound data), but for 16 layer VGGNet with several hundreds millions of parameters, if one doesn't pay attention on how the memory is assigned and managed, it can eat up entire memory easily. Python tends not to release memories assigned for variables. Python is supposed to manage memory in a smart way automatically, but most of time, it just doesn't release. Since the name of the variable in python is just a pointer, not the object itself, there are variable values persist sitting in the memory even when I update the values for certain variable name (Some people say they had a luck when they specify some options in keras backend module K, but those didn't work in my case). The tensorflow shouldn't have this problem due to its graph computation scheme, but with interactions with keras it seemed something is not working as it would naturally do in tensorflow. So I failed to load VGGNet without any data yet in keras+tensorflow.
I had a choice to use theano instead, but I decided to explore tensorflow without keras.  

#### Pre-trained VGGNet models:  
[VGG16 in keras](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)   
[VGG16 in tensorflow](https://github.com/machrisaa/tensorflow-vgg)

## Understanding how tensorflow works (graph computation and graph visualization)
Very useful tensorflow tutorial: https://www.oreilly.com/learning/hello-tensorflow

## Progress
#### some time ago
modvgg16.py: Modified pre-trained VGGNet code to become CNN-code and modified dimension of fc8 layer (output).
Tested on sample images using sysnet labels which is not meaningful (by default label 0 is fish) but just shows that the modified VGGNet is working and gives output.
It does not have training part yet.

#### 08-30-2016
#### Test loading 25000 images in numpy array
Still working on loading training data issue.
Tried np.zero((25000,224,224,3)) approach where I assign array then run a for loop to fill the array.
In principle, this should not creep up the memory but it did. It was a little better in that I was able to load all training images in one array in the end (only when I reduce the data type from float64 to flaot32), but it almost maxed out the memory and the after was pretty much useless since the computer is very slow to do anything further.

I tried np.empty plus np.concatenate approach.
It is extremely slow when it does concat operation in every loop.
I can see the memory jumps up and down, so python is doing some memory cleaning but it slowly creeps up too, and eventually it maxed out the memory without finishing loading- so it seems not a good approach.

In conclusion, I need a batch loading rather than loading all of data.

#### Inspecting how MNIST next_batch works:
>In [6]: im, la = callbatch(path,1000)   
1000    
In [7]: test = DataSet(im, la)   
In [8]: imbat0, labat0 = test.next_batch(100)    
In [9]: imbat0.shape    
Out[9]: (100, 224, 224, 3)   
In [10]: labat0.shape   
Out[10]: (100, 2)

It seems that MNIST next_batch function takes entire image & label array and then makes slices of them. MNIST data has smaller number and dimension (10000 x 28 x28 x1) compared to my dataset in ImageNet form (25000 x 224 x 224 x 3). Also MNIST data is in binary format than my dataset in jpeg format, so I'm not sure how the memory management are done differently when loading images in different format. I could try looking what data loading method CIFAR10 example is using, but the size/dimension of CIFAR10 is not very different from MNIST (except that CIFAR10 has color), certainly ImageNet may need a different approach.

So I implemented a function which takes a list of file names as input and get the images and labels for outputs inside the class DataSet. This modified class shuffles file name list every epoch, then prepares images and labels by loading the files as batch for each batch inside an epoch.

#### training module
see https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/mnist/fully_connected_feed.py

#### 09-01-2016
Worked on training module to work.
Referenced MNIST example from TensorFlow website.
tf.flag doesn't work well with custom class, so I rewrote the main function as class Training().
It took a while to understand what the example code does. The main difference is the model building process where mine is building model as an object (I'm not sure if it's a part of reason for the memory issue below).
The training worked in the end, but when I used batch_size of 100, gtx 970 failed to load the graph- which I don't understand why tensorflow has to pre-allocate 100 times of all vgg weights all at once. In the end I used smaller batch_size 10, and smaller input data number to begin with, and it did work. I still need to organize other data set such as validation and test in order to generate the report and evaluation.

#### 09-11-2016
Modified the run_training function and made the training, validation, and test evaluations to work. The test dataset from kaggle data did not have label (since the competition evaluates on test set internally), so I ended up not using them, but random sampled from training data and set aside- same procedure for validation set.
My code can limit how much total data to use- for now I tested with 1000 total data (train+validation+test) with default 0.7 train-test split ratio, then train data is further divided by default 0.8 train-validation split ratio. I tried batch size of 100, 50 and 10 but the computer resource seems to handle only small batch size of 10 (did not accept 50). For this input with default n_epoch = 5 and learning rate 0.01, I got 96% of precision for cat vs dog image data, and it took just a few seconds. 

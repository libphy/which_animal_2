from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import modvgg16
import load_data
import imageprep
import time



class Training(object):
    def __init__(self, path_train, path_test, train_load_limit=None, batch_size=100, n_epoch=5, learning_rate = 0.01, train_val_ratio = 0.8):
        self.train_dir = path_train
        self.test_dir = path_test
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.train_val_ratio = train_val_ratio
        if train_load_limit == None:
            self.max_num_train = 25000
        else:
            self.max_num_train = train_load_limit
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_string('train_dir', '/home/geena/projects/which_animal/data/images/train/processed/', 'Directory to put the training data.')

    def placeholder_inputs(self):
        images_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, 224,224,3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size))
        return images_placeholder, labels_placeholder

    def fill_feed_dict(self, data_obj, images_pl, labels_pl):
        images_feed, labels_feed = data_obj.next_batch(self.batch_size)
        feed_dict = {
            images_pl: images_feed,
            labels_pl: labels_feed,
        }
        return feed_dict

    def do_eval(self, sess, eval_correct, images_placeholder, labels_placeholder, data_obj):
        true_count = 0 #number of correct predictions
        steps_per_epoch = data_obj.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        for step in xrange(steps_per_epoch):
            feed_dict = self.fill_feed_dict(data_obj, images_placeholder, labels_placeholder)
            true_count += sess.run(eval_correct, feed_dict = feed_dict)
        precision = true_count/num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %(num_examples, true_count, precision))

    def loss(self, logits, labels):
        """Calculates the loss from the logits and the labels.
        Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
        Returns:
        loss: Loss tensor of type float.
        """
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits, labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def training(self, loss, learning_rate):
        """Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
        Returns:
        train_op: The Op for training.
        """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(loss.op.name, loss)
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def evaluation(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
          range [0, NUM_CLASSES).
        Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
        """
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        correct = tf.nn.in_top_k(logits, labels, 1)
        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))


    def run_training(self):
        """Train MNIST for a number of steps."""
        # Get the sets of images and labels for training, validation, and
        # test on MNIST.
        inputfiles = os.listdir(self.train_dir)
        np.random.shuffle(inputfiles)
        trainfiles = inputfiles[:self.max_num_train]
        testlist = os.listdir(self.test_dir)
        np.random.shuffle(testlist)
        trainlist = trainfiles[:int(len(trainfiles)*self.train_val_ratio)]
        validationlist = trainfiles[int(len(trainfiles)*self.train_val_ratio):]
        trainset = load_data.DataSet(trainlist,self.train_dir)
        validationset = load_data.DataSet(validationlist,self.train_dir)
        testset = load_data.DataSet(testlist,self.test_dir)
        print('Train list: ', len(trainlist))
        print('Validation list: ', len(validationlist))
        print('Test list: ', len(testlist))

        with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
            # Generate placeholders for the images and labels.
            images_placeholder, labels_placeholder = self.placeholder_inputs()
            # Build a Graph that computes predictions from the inference model.
            vgg = modvgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images_placeholder)
            #init_op = tf.initialize_all_variables()
            #sess.run(init_op)
            #logits = sess.run(vgg.prob, feed_dict=feed_dict)
            logits = vgg.prob
            # Add to the Graph the Ops for loss calculation.
            print(logits.get_shape())
            print(labels_placeholder.get_shape())
            loss = self.loss(logits, labels_placeholder)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = self.training(loss, self.learning_rate)

            # Add the Op to compare the logits to the labels during evaluation.
            eval_correct = self.evaluation(logits, labels_placeholder)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()

            # Add the variable initializer Op.
            init = tf.initialize_all_variables()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.train.SummaryWriter('log', sess.graph)

            # And then after everything is built:

            # Run the Op to initialize the variables.
            sess.run(init)

            # Start the training loop.
            for step in xrange(self.n_epoch * self.batch_size):
                start_time = time.time()

              # Fill a feed dictionary with the actual set of images and labels
              # for this particular training step.
                feed_dict = self.fill_feed_dict(trainset,
                                         images_placeholder,
                                         labels_placeholder)

              # Run one step of the model.  The return values are the activations
              # from the `train_op` (which is discarded) and the `loss` Op.  To
              # inspect the values of your Ops or variables, you may include them
              # in the list passed to sess.run() and the value tensors will be
              # returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss],
                                       feed_dict=feed_dict)

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % self.batch_size == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    # Update the events file.
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % self.batch_size == 0 or (step + 1) == self.max_num_train:
                    checkpoint_file = os.path.join('log', 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.
                    print('Training Data Eval:')
                    self.do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            trainset)
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    self.do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            validationset)
                    # Evaluate against the test set.
                    print('Test Data Eval:')
                    self.do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            testset)


if __name__ == '__main__':

# #the images are already scaled
    path_train = '/home/geena/projects/which_animal_2/data/train/processed/'
    path_test = '/home/geena/projects/which_animal_2/data/test/processed/'
    if not os.path.isdir(path_train):
        os.mkdir(path_train)
        imageprep.onebyone('/home/geena/projects/which_animal_2/data/train/')
    if not os.path.isdir(path_test):
        os.mkdir(path_test)
        imageprep.onebyone('/home/geena/projects/which_animal_2/data/test/')
    train = Training(path_train, path_test, 100, batch_size=10)
    train.run_training()

from datetime import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image

from network import *
from utils import ImageReader, decode_labels, inv_preprocess, prepare_label, write_log, read_labeled_image_list



"""
This script trains or evaluates the model on augmented PASCAL VOC 2012 dataset.
The training set contains 10581 training images.
The validation set contains 1449 validation images.

Training:
'poly' learning rate
different learning rates for different layers
"""



IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class Model(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.conf.top_scope = tf.get_variable_scope()

    # train
    def train(self):
        self.train_setup()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())


        if self.conf.start_step == 0:
            # Load the pre-trained model if provided
            if self.conf.checkpoint_file is not None:
                self.load(self.loader, self.conf.pretrain_file)
            pass
        else:
            # Load the checkpoint of model if provided
            if self.conf.checkpoint_file is not None:
                self.load(self.loader_checkpoint, self.conf.checkpoint_file)


        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        # Train!
        c_gradients =  [0, 0, 0]
        c_vector = np.array([0, 0, 0])
        c_1 = np.array([0])
        c_2 = np.array([0])
        c_3 = np.array([0])
        for step in range(self.conf.start_step, self.conf.start_step+self.conf.num_steps+1):
            start_time = time.time()
            feed_dict = { self.curr_step : step }
            mIoU = 0

            if step % self.conf.save_interval == 0:
#                loss_value, images, labels, preds, summary, _ = self.sess.run(
#                    [self.reduced_loss,
#                    self.image_batch,
#                    self.label_batch,
#                    self.pred,
#                    self.total_summary,
#                    self.train_op],
#                    feed_dict=feed_dict)
#                loss_value, c_gradients, _, _ = self.sess.run([self.reduced_loss, self.c_gradients,
                loss_value, _, _ = self.sess.run([self.reduced_loss,
                    self.train_op,
                    self.mIou_update_op],
                    feed_dict=feed_dict)

                mIoU = self.mIoU.eval(session=self.sess)
                summary = tf.Summary()
                summary.value.add(tag='mIoU', simple_value=mIoU)
                summary.value.add(tag='loss', simple_value=loss_value)

                try:
#                    var_c = [v for v in tf.global_variables() if "c_1" in v.name][0]
#                    c_1 = var_c.eval(session=self.sess)
#                    var_c = [v for v in tf.global_variables() if "c_2" in v.name][0]
#                    c_2 = var_c.eval(session=self.sess)
#                    var_c = [v for v in tf.global_variables() if "c_3" in v.name][0]
#                    c_3 = var_c.eval(session=self.sess)



                    var_c = [v for v in tf.global_variables() if "c_vector" in v.name][0]
                    c_vector = var_c.eval(session=self.sess)
                    summary = tf.Summary()
                    summary.value.add(tag='C_1', simple_value=var_c[0])
                    summary.value.add(tag='C_2', simple_value=var_c[1])
                    summary.value.add(tag='C_3', simple_value=var_c[2])
                    self.summary_writer_train.add_summary(summary,
                                                         (self.conf.start_step+self.conf.num_steps))
                except:
                    pass

                self.summary_writer_train.add_summary(summary, step)
#                self.save(self.saver, step)
            else:
#                loss_value, c_gradients,  _, _ = self.sess.run([self.reduced_loss, self.c_gradients,
                loss_value,  _, _ = self.sess.run([self.reduced_loss,
                                                     self.train_op, self.mIou_update_op], feed_dict=feed_dict)
                mIoU = self.mIoU.eval(session=self.sess)
                try:
#                    var_c = [v for v in tf.global_variables() if "c_1" in v.name][0]
#                    c_1 = var_c.eval(session=self.sess)
#                    var_c = [v for v in tf.global_variables() if "c_2" in v.name][0]
#                    c_2 = var_c.eval(session=self.sess)
#                    var_c = [v for v in tf.global_variables() if "c_3" in v.name][0]
#                    c_3 = var_c.eval(session=self.sess)



                    var_c = [v for v in tf.global_variables() if "c_vector" in v.name][0]
                    c_vector = var_c.eval(session=self.sess)
                    summary = tf.Summary()
                    summary.value.add(tag='C_1', simple_value=var_c[0])
                    summary.value.add(tag='C_2', simple_value=var_c[1])
                    summary.value.add(tag='C_3', simple_value=var_c[2])
                    self.summary_writer_train.add_summary(summary,
                                                         (self.conf.start_step+self.conf.num_steps))
                except:
                    pass

            duration = time.time() - start_time
#           print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
            write_log('{:d}, {:.3f}, {:.3f}, {:}, [{:.3f}, {:.3f}, {:.3f}]'.format(step, loss_value, mIoU, c_gradients, c_vector[0], c_vector[1], c_vector[2] ), self.conf.logfile)
#            write_log('{:d}, {:.3f}, {:.3f}, [{:}, {:}, {:}]'.format(step, loss_value, mIoU, str(c_1), str(c_2), str(c_3)), self.conf.logfile)

        # finish
        self.save(self.saver, step)
        self.coord.request_stop()
        self.coord.join(threads)

    # evaluate
    def test(self):
        self.test_setup()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # load checkpoint
        checkpointfile = self.conf.modeldir+ '/model.ckpt-' + str(self.conf.valid_step)
        self.load(self.loader, checkpointfile)
        
        
        try:
            var_c = [v for v in tf.global_variables() if "c_vector" in v.name][0]
            summary = tf.Summary()
            summary.value.add(tag='C_1', simple_value=var_c[0])
            summary.value.add(tag='C_2', simple_value=var_c[1])
            summary.value.add(tag='C_3', simple_value=var_c[2])
            self.summary_writer_test.add_summary(summary,
                                                 (self.conf.start_step+self.conf.num_steps))
        except:
            pass
        
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        # Test!
        confusion_matrix = np.zeros((self.conf.num_classes, self.conf.num_classes), dtype=np.int)
        for step in range(self.conf.valid_num_steps):
            preds, _, _, c_matrix = self.sess.run([self.pred, self.accu_update_op, self.mIou_update_op, self.confusion_matrix])
            confusion_matrix += c_matrix
#            if step % 100 == 0:
#                write_log('step {:d}'.format(step), self.conf.logfile)
        accuracy = self.accu.eval(session=self.sess)

        write_log('Pixel Accuracy: {:.3f}'.format(accuracy), self.conf.logfile)
#        write_log('Mean IoU: {:.3f}'.format(self.mIoU.eval(session=self.sess)), self.conf.logfile)
        summary = self.compute_IoU_per_class(confusion_matrix)
        summary.value.add(tag='pixel accuracy', simple_value=accuracy)
        self.summary_writer_test.add_summary(summary,
                                             (self.conf.start_step+self.conf.num_steps))
        

        # finish
        self.coord.request_stop()
        self.coord.join(threads)

    # prediction
    def predict(self):
        self.predict_setup()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # load checkpoint
        checkpointfile = self.conf.modeldir+ '/model.ckpt-' + str(self.conf.valid_step)
        self.load(self.loader, checkpointfile)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        # img_name_list
        image_list, _ = read_labeled_image_list('', self.conf.test_data_list)

        # Predict!
        for step in range(self.conf.test_num_steps):
            preds = self.sess.run(self.pred)

            img_name = image_list[step].split('/')[2].split('.')[0]
            # Save raw predictions, i.e. each pixel is an integer between [0,20].
            im = Image.fromarray(preds[0,:,:,0], mode='L')
            filename = '/%s_mask.png' % (img_name)
            im.save(self.conf.out_dir + '/prediction' + filename)

            # Save predictions for visualization.
            # See utils/label_utils.py for color setting
            # Need to be modified based on datasets.
            if self.conf.visual:
                msk = decode_labels(preds, num_classes=self.conf.num_classes)
                im = Image.fromarray(msk[0], mode='RGB')
                filename = '/%s_mask_visual.png' % (img_name)
                im.save(self.conf.out_dir + '/visual_prediction' + filename)

            if step % 100 == 0:
                write_log('step {:d}'.format(step), self.conf.logfile)

        write_log('The output files has been saved to {}'.format(self.conf.out_dir), self.conf.logfile)

        # finish
        self.coord.request_stop()
        self.coord.join(threads)

    def train_setup(self):
        tf.set_random_seed(self.conf.random_seed)
        
        # Create queue coordinator.
        self.coord = tf.train.Coordinator()

        # Input size
        input_size = (self.conf.input_height, self.conf.input_width)
        
        # Load reader
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                self.conf.data_dir,
                self.conf.data_list,
                input_size,
                self.conf.random_scale,
                self.conf.random_mirror,
                self.conf.ignore_label,
                IMG_MEAN,
                self.coord)
            self.image_batch, self.label_batch = reader.dequeue(self.conf.batch_size)
        
        # Create network
        if self.conf.encoder_name not in ['res101', 'res50', 'deeplab']:
            print('encoder_name ERROR!')
            print("Please input: res101, res50, or deeplab")
            sys.exit(-1)
        elif self.conf.encoder_name == 'deeplab':
            net = Deeplab_v2(self.image_batch, self.conf.num_classes, True,
                             self.conf.dilated_type, self.conf.top_scope)
            # Variables that load from pre-trained model.
            restore_var = [v for v in tf.global_variables() if 'fc' not in v.name
                           and 'fix_w' not in v.name and 'w_avg' not in v.name
                           and 'w_gauss' not in v.name and 'c_' not in v.name
                           and 'gauss_sigma' not in v.name ]
            # Trainable Variables
            all_trainable = tf.trainable_variables()
            # Fine-tune part
            encoder_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
            # Decoder part
#            decoder_trainable = [v for v in all_trainable if 'fc' in v.name and
            decoder_trainable = [v for v in all_trainable if 'fc' in v.name and
                                 'w_avg' not in v.name and 'w_gauss' not in v.name
                                 and 'gauss_sigma' not in v.name]
        else:
            net = ResNet_segmentation(self.image_batch, self.conf.num_classes, True, self.conf.encoder_name, self.conf.dilated_type, self.conf.top_scope)
            # Variables that load from pre-trained model.
            restore_var = [v for v in tf.global_variables() if 'resnet_v1' in v.name and 'fix_w' not in v.name]
            # Trainable Variables
            all_trainable = tf.trainable_variables()
            # Fine-tune part
            encoder_trainable = [v for v in all_trainable if 'resnet_v1' in v.name] # lr * 1.0
            # Decoder part
            decoder_trainable = [v for v in all_trainable if 'decoder' in v.name]
        
        decoder_w_trainable = [v for v in decoder_trainable if 'weights' in v.name or 'gamma' in v.name] # lr * 10.0
        decoder_b_trainable = [v for v in decoder_trainable if 'biases' in v.name or 'beta' in v.name] # lr * 20.0

#        decoder_pre_trainable = [v for v in all_trainable if 'c_vector' in v.name or 'gauss_sigma' in v.name] # lr * 10.0
        decoder_pre_trainable = [v for v in all_trainable if 'c_vector' in v.name] # lr * 10.0
#        assert(len(decoder_pre_trainable) > 0 )
        # Check
        assert(len(all_trainable) == len(decoder_trainable) + len(encoder_trainable))
        assert(len(decoder_trainable) == len(decoder_w_trainable) + len(decoder_b_trainable))

        # Network raw output
        raw_output = net.outputs # [batch_size, h, w, 21]

        # Output size
        output_shape = tf.shape(raw_output)
        output_size = (output_shape[1], output_shape[2])

        # Groud Truth: ignoring all labels greater or equal than n_classes
        label_proc = prepare_label(self.label_batch, output_size, num_classes=self.conf.num_classes, one_hot=False)
        raw_gt = tf.reshape(label_proc, [-1,])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.conf.num_classes - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        raw_prediction = tf.reshape(raw_output, [-1, self.conf.num_classes])
        prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax_cross_entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        # L2 regularization
        l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in all_trainable if 'weights' in v.name]
        # Loss function
        self.reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

        # Define optimizers
        # 'poly' learning rate
        base_lr = tf.constant(self.conf.learning_rate)
        self.curr_step = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.curr_step / self.conf.max_steps), self.conf.power))
        # We have several optimizers here in order to handle the different lr_mult
        # which is a kind of parameters in Caffe. This controls the actual lr for each
        # layer.
        opt_encoder = tf.train.MomentumOptimizer(learning_rate, self.conf.momentum)
        opt_decoder_w = tf.train.MomentumOptimizer(learning_rate * 10.0, self.conf.momentum)
        opt_decoder_b = tf.train.MomentumOptimizer(learning_rate * 20.0, self.conf.momentum)
        opt_decoder_pre = tf.train.MomentumOptimizer(learning_rate * 2, self.conf.momentum)

        # To make sure each layer gets updated by different lr's, we do not use 'minimize' here.
        # Instead, we separate the steps compute_grads+update_params.
        # Compute grads
        grads = tf.gradients(self.reduced_loss, encoder_trainable + decoder_w_trainable + decoder_b_trainable + decoder_pre_trainable )
        grads_encoder = grads[:len(encoder_trainable)]
        grads_decoder_w = grads[len(encoder_trainable) : (len(encoder_trainable) + len(decoder_w_trainable))]
        grads_decoder_b = grads[(len(encoder_trainable) + len(decoder_w_trainable)) : (len(encoder_trainable) + len(decoder_w_trainable) + len(decoder_b_trainable))]
        grads_decoder_pre = grads[(len(encoder_trainable) + len(decoder_w_trainable) + len(decoder_b_trainable)):]


#
#        grads_decoder_pre = [tf.clip_by_value(grad, -5., 5.) for grad in grads_decoder_pre]
#
        self.c_gradients = grads_decoder_pre
        

        # Update params
        train_op_conv = opt_encoder.apply_gradients(zip(grads_encoder, encoder_trainable))
        train_op_fc_w = opt_decoder_w.apply_gradients(zip(grads_decoder_w, decoder_w_trainable))
        train_op_fc_b = opt_decoder_b.apply_gradients(zip(grads_decoder_b, decoder_b_trainable))


#
#        train_op_fc_pre = opt_decoder_pre.apply_gradients(zip(grads_decoder_pre, decoder_pre_trainable))
#        
        # Finally, get the train_op!
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for collecting moving_mean and moving_variance
        with tf.control_dependencies(update_ops):
            
##            self.train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b, train_op_fc_pre)
            self.train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

        # Saver for storing checkpoints of the model
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

        # Loader for loading the pre-trained model
        self.loader = tf.train.Saver(var_list=restore_var)

        # Loader for loading the checkpoint files
        self.loader_checkpoint = tf.train.Saver(var_list=tf.global_variables())

        # Training summary
        # Processed predictions: for visualisation.
        raw_output_up = tf.image.resize_bilinear(raw_output, input_size)
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        self.pred = tf.expand_dims(raw_output_up, dim=3)

         #mIoU
        pred_logits = tf.reshape(self.pred, [-1, ])
        gt = tf.reshape(self.label_batch, [-1, ])
         # Ignoring all labels greater than or equal to n_classes.
        temp = tf.less_equal(gt, self.conf.num_classes - 1)
        weights = tf.cast(temp, tf.int32)
        # fix for tf 1.3.0
        gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))

        # mIoU
        self.mIoU, self.mIou_update_op = tf.contrib.metrics.streaming_mean_iou(pred_logits, gt, num_classes=self.conf.num_classes, weights=weights)

        # Pixel accuracy
        self.accu, self.accu_update_op = tf.contrib.metrics.streaming_accuracy(
            pred_logits, gt, weights=weights)

        # Image summary.
#        images_summary = tf.py_func(inv_preprocess, [self.image_batch, 2, IMG_MEAN], tf.uint8)
#        labels_summary = tf.py_func(decode_labels, [self.label_batch, 2, self.conf.num_classes], tf.uint8)
#        preds_summary = tf.py_func(decode_labels, [self.pred, 2, self.conf.num_classes], tf.uint8)
#        self.total_summary = tf.summary.image('images',
#            tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
#            max_outputs=2) # Concatenate row-wise.


        # Add Training summary
#        tf.summary.scalar('loss', self.reduced_loss)
#        tf.summary.scalar('pixel_accuracy', self.accu)
#        tf.summary.scalar('mIoU', self.mIoU)
        self.total_summary = tf.summary.merge_all()

        if not os.path.exists(self.conf.logdir):
            os.makedirs(self.conf.logdir)
        self.summary_writer_train = tf.summary.FileWriter(self.conf.logdir+'/train', graph=tf.get_default_graph())

    def test_setup(self):
        # Create queue coordinator.
        self.coord = tf.train.Coordinator()

        # Load reader
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                self.conf.data_dir,
                self.conf.valid_data_list,
                None, # the images have different sizes
                False, # no data-aug
                False, # no data-aug
                self.conf.ignore_label,
                IMG_MEAN,
                self.coord)
            image, label = reader.image, reader.label # [h, w, 3 or 1]
        # Add one batch dimension [1, h, w, 3 or 1]
        self.image_batch, self.label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)
        
        # Create network
        if self.conf.encoder_name not in ['res101', 'res50', 'deeplab']:
            print('encoder_name ERROR!')
            print("Please input: res101, res50, or deeplab")
            sys.exit(-1)
        elif self.conf.encoder_name == 'deeplab':
            net = Deeplab_v2(self.image_batch, self.conf.num_classes, False, self.conf.dilated_type, self.conf.top_scope)
        else:
            net = ResNet_segmentation(self.image_batch, self.conf.num_classes, False, self.conf.encoder_name, self.conf.dilated_type, self.conf.top_scope)

        # predictions
        raw_output = net.outputs
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(self.image_batch)[1:3,])
        raw_output = tf.argmax(raw_output, axis=3)
        pred = tf.expand_dims(raw_output, dim=3)
        self.pred = tf.reshape(pred, [-1,])
        # labels
        gt = tf.reshape(self.label_batch, [-1,])
        # Ignoring all labels greater than or equal to n_classes.
        temp = tf.less_equal(gt, self.conf.num_classes - 1)
        weights = tf.cast(temp, tf.int32)

        # fix for tf 1.3.0
        gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))

        # Pixel accuracy
        self.accu, self.accu_update_op = tf.contrib.metrics.streaming_accuracy(
            self.pred, gt, weights=weights)

        # mIoU
        self.mIoU, self.mIou_update_op = tf.contrib.metrics.streaming_mean_iou(
            self.pred, gt, num_classes=self.conf.num_classes, weights=weights)

        # confusion matrix
        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(
            self.pred, gt, num_classes=self.conf.num_classes, weights=weights)

        # Loader for loading the checkpoint
        self.loader = tf.train.Saver(var_list=tf.global_variables())
        self.summary_writer_test = tf.summary.FileWriter(self.conf.logdir+'/test')

    def predict_setup(self):
        # Create queue coordinator.
        self.coord = tf.train.Coordinator()

        # Load reader
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                self.conf.data_dir,
                self.conf.test_data_list,
                None, # the images have different sizes
                False, # no data-aug
                False, # no data-aug
                self.conf.ignore_label,
                IMG_MEAN,
                self.coord)
            image, label = reader.image, reader.label # [h, w, 3 or 1]
        # Add one batch dimension [1, h, w, 3 or 1]
        image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)

        # Create network
        if self.conf.encoder_name not in ['res101', 'res50', 'deeplab']:
            print('encoder_name ERROR!')
            print("Please input: res101, res50, or deeplab")
            sys.exit(-1)
        elif self.conf.encoder_name == 'deeplab':
            net = Deeplab_v2(image_batch, self.conf.num_classes, False, self.conf.dilated_type, self.conf.top_scope)
        else:
            net = ResNet_segmentation(image_batch, self.conf.num_classes, False, self.conf.encoder_name, self.conf.dilated_type, self.conf.top_scope)

        # Predictions.
        raw_output = net.outputs
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
        raw_output = tf.argmax(raw_output, axis=3)
        self.pred = tf.cast(tf.expand_dims(raw_output, dim=3), tf.uint8)

        # Create directory
        if not os.path.exists(self.conf.out_dir):
            os.makedirs(self.conf.out_dir)
            os.makedirs(self.conf.out_dir + '/prediction')
            if self.conf.visual:
                os.makedirs(self.conf.out_dir + '/visual_prediction')

        # Loader for loading the checkpoint
        self.loader = tf.train.Saver(var_list=tf.global_variables())

    def save(self, saver, step):
        '''
        Save weights.
        '''
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(self.conf.modeldir, model_name)
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        saver.save(self.sess, checkpoint_path, global_step=step)
        write_log('The checkpoint has been created.', self.conf.logfile)

    def load(self, saver, filename):
        '''
        Load trained weights.
        ''' 
        saver.restore(self.sess, filename)
        write_log("Restored model parameters from {}".format(filename), self.conf.logfile)

    def compute_IoU_per_class(self, confusion_matrix):
        mIoU = 0
        summary = tf.Summary()
        for i in range(self.conf.num_classes):
            # IoU = true_positive / (true_positive + false_positive + false_negative)
            TP = confusion_matrix[i,i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i]) - TP
            IoU = TP / (TP + FP + FN)
            write_log ('class %d: %.3f' % (i, IoU), self.conf.logfile)
            summary.value.add(tag='IoU class %d' % (i), simple_value=IoU)
            mIoU += IoU / self.conf.num_classes
        write_log ('mIoU: %.3f' % mIoU, self.conf.logfile)
        summary.value.add(tag='mIoU', simple_value=mIoU)
        return summary 

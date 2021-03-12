'''
    PointNet version 1 Model
    Reference: https://github.com/charlesq34/pointnet
'''
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    labels_pl_b = tf.placeholder(tf.int32, shape=(batch_size))
    lam = tf.placeholder(tf.float32, shape=(batch_size))
    return pointclouds_pl, labels_pl, labels_pl_b, lam


def get_model(point_cloud, is_training, bn_decay=None, class_num=40):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)
    
    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    
    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    if class_num==10:
        net = tf_util.fully_connected(net, 10, activation_fn=None, scope='fc3') # for ModelNet10
    else:
        net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
        
    return net, end_points


def get_loss(pred, label, end_points, label_b, lam):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    # classify_loss = tf.reduce_mean(loss)
    # tf.summary.scalar('classify loss', classify_loss)
    # tf.add_to_collection('losses', classify_loss)
    loss_a_lam = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)*(1-lam)
    loss_b_lam = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label_b)*lam
    loss_sum = tf.add(loss_a_lam, loss_b_lam)
    
    classify_loss = tf.reduce_mean(loss_sum)
    
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)

import tensorflow.keras.backend as K
import tensorflow as tf
from n2v.internals.n2v_losses import loss_mse as n2v_mseloss
from n2v.internals.n2v_losses import loss_mae as n2v_maeloss

from tensorflow.nn import softmax_cross_entropy_with_logits as cross_entropy
from .seg_losses import asym_unified_focal_loss as aufl


def loss_denoiseg(alpha=0.5, loss = 2, relative_weights=[1.0, 1.0, 5.0],num_class=10):
    """
    Calculate DenoiSeg loss which is a weighted sum of segmentation- and
    noise2void-loss

    :param lambda_: relative weighting, 0 means denoising, 1 means segmentation; (Default: 0.5)
    :param relative_weights: Segmentation class weights (background, foreground, border); (Default: [1.0, 1.0, 5.0])
    :return: DenoiSeg loss
    """

    denoise_loss = denoiseg_denoise_lossx(weight=alpha,loss= loss, num_class=num_class)
    # seg_loss = eval('denoiseg_seg_lossx(weight=(1 - alpha), relative_weights=relative_weights)
    seg_loss = denoiseg_seg_lossx(weight=(1 - alpha), relative_weights=relative_weights,num_class=num_class)


    def denoiseg(y_true, y_pred):
        return seg_loss(y_true, y_pred) + denoise_loss(y_true, y_pred)

    return denoiseg

def denoiseg_seg_lossx(num_class,weight=0.5, relative_weights=[1.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0],):
    class_weights = tf.constant([relative_weights])

    def seg_loss(y_true, y_pred):
        gt_channel_axis = len(y_true.shape) - 1
        target, mask, *bg_l9_gt = tf.split(y_true, num_class+2, axis=gt_channel_axis)

        pred_channel_axis = len(y_pred.shape) - 1
        denoised, *bg_l9_pred = tf.split(y_pred, num_class+1, axis=pred_channel_axis)

        onehot_gt = tf.reshape(tf.stack(bg_l9_gt, axis=pred_channel_axis), [-1, num_class])
        weighted_gt = tf.reduce_sum(class_weights * onehot_gt, axis=1)

        onehot_pred = tf.reshape(tf.stack(bg_l9_pred, axis=pred_channel_axis), [-1, num_class])

        segmentation_loss = K.mean(
            tf.reduce_sum(onehot_gt, axis=-1) * (cross_entropy(logits=onehot_pred, labels=onehot_gt) * weighted_gt)
        )

        return weight * segmentation_loss

    return seg_loss

def denoiseg_denoise_lossx(weight=0.5,loss = 2, num_class = 10):
    if loss == 2:
        n2v_loss = n2v_mseloss()
    elif loss == 1:
        n2v_loss = n2v_maeloss()


    def denoise_loss(y_true, y_pred):
        channel_axis = len(y_true.shape) - 1

        target, mask, *bg_l9_gt = tf.split(y_true, num_class+2, axis=channel_axis)
    # denoised, pred_bg, pred_fg, pred_b = tf.split(y_pred, 4, axis=len(y_pred.shape) - 1)
        denoised, *bg_l9_pred = tf.split(y_pred, num_class+1, axis=len(y_pred.shape) - 1)

        return weight * n2v_loss(tf.concat([target, mask], axis=channel_axis), denoised)

    return denoise_loss

import numpy as np
from sklearn.feature_extraction import image
# import albumentations as A



def create_patches(images, masks, size):
    """
    Creates square patches from images and masks of a specified `size`.

    Parameters
    ----------
    images : array(float)
        Array of images.
    masks : array(int)
        Array of labelled images.
    size: int
        Width of the patch
    Returns
    -------
    patchesimages : array(float)
        Array of training patches.
    patchesmasks : array(float)
        Array of labelled training patches.
    """

    patchesimages = image.extract_patches_2d(images, (size, size), max_patches=10, random_state=0)
    patchesmasks = image.extract_patches_2d(masks, (size, size), max_patches=10, random_state=0)
    return patchesimages, patchesmasks


def combine_train_test_data(X_train, Y_train, X_test, Y_test):
    """
    Combines train and test data along 0th dimension.

    Parameters
    ----------
    X_train : array(float)
        Array of training images.
    Y_train : float
        Array of labelled training images.
    X_test : array(float)
        Array of test images.
    Y_test : float
        Array of labelled test images.
    Returns
    -------
    X_train_N2V : array(float)
        Combined array of training images.
    Y_train_N2V : array(float)
        Combined array of labelled training images.
    """

    if (X_test.ndim == X_train.ndim):
        if (X_test.shape[1] == X_train.shape[1] and X_test.shape[2] == X_train.shape[2]):
            X_test_patches = X_test
            Y_test_patches = Y_test
    else:
        X_test_patches, Y_test_patches = create_patches(X_test[0], Y_test[0], X_train.shape[1])
        for image_num in range(1, X_test.shape[0]):
            patchesimages, patchesmasks = create_patches(X_test[image_num], Y_test[image_num], X_train.shape[1])
            X_test_patches = np.concatenate((X_test_patches, patchesimages))
            Y_test_patches = np.concatenate((Y_test_patches, patchesmasks))

    X_train_N2V = np.concatenate((X_train, X_test_patches))
    Y_train_N2V = np.concatenate((Y_train, Y_test_patches))

    return X_train_N2V, Y_train_N2V


def shuffle_train_data(X_train, Y_train, random_seed):
    """
    Shuffles data with seed 1.

    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    Returns
    -------
    X_train : array(float)
        shuffled array of training images.
    Y_train : array(float)
        Shuffled array of labelled training images.
    """
    np.random.seed(random_seed)
    seed_ind = np.random.permutation(X_train.shape[0])
    X_train = X_train[seed_ind]
    Y_train = Y_train[seed_ind]

    return X_train, Y_train


def augment_data(X_train, Y_train):
    """
    Augments the data 8-fold by 90 degree rotations and flipping.

    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    Returns
    -------
    X_train_aug : array(float)
        Augmented array of training images.
    Y_train_aug : array(float)
        Augmented array of labelled training images.
    """
    X_ = X_train.copy()

    # X_train_aug = np.concatenate((X_train, np.rot90(X_, 1, (1, 2))))
    # X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 2, (1, 2))))
    # X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 3, (1, 2))))
    # X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1)))
    X_train_aug = np.concatenate((X_train, np.flip(X_, axis=2)))
    # X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1)))

    Y_ = Y_train.copy()

    # Y_train_aug = np.concatenate((Y_train, np.rot90(Y_, 1, (1, 2))))
    # Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 2, (1, 2))))
    # Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 3, (1, 2))))
    # Y_train_aug = np.concatenate((Y_train_aug, np.flip(Y_train_aug, axis=1)))

    Y_train_aug = np.concatenate((Y_train, np.flip(Y_, axis=2)))
    # Y_train_aug = np.concatenate((Y_train_aug, np.flip(Y_train_aug, axis=1)))

    print('Raw image size after augmentation', X_train_aug.shape)
    print('Mask size after augmentation', Y_train_aug.shape)

    return X_train_aug, Y_train_aug


def touint8(img):
    img = (img - img.min()) / (img.max() - img.min())
    img *= 255.0
    img = img.astype('uint8')
    return img


# def onehotfrompixels(label):
#     for i in range(label.shape[0]):
#         label_ = np.zeros([label.shape[0],label.shape[1],10])

#         for layer in range(10):
#             label_[:,:,layer] = label==layer

#     return label_

# def super_augment_data(X_train, Y_train):
#     transform = A.Compose(
#         [
#             # A.CLAHE(p=0.0),
#         #  A.HorizontalFlip(p=0.5),
#          #  A.RandomRotate90(),
#          #  A.Transpose(),
#          #  A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0,
#          #                     rotate_limit=45, p=1),
#          #  A.Blur(blur_limit=3,p=1),
#          #  A.OpticalDistortion(distort_limit= (-0.6,-0.2),p=1),
#          #  A.ElasticTransform(sigma =, p=1),
#          A.GridDistortion(distort_limit=(-0.3, 0.3), p=0.5),
#          #  A.Affine(p=1)
#          #  A.HueSaturationValue(p=1)
#          ])
#     augmented_images = np.zeros(X_train.shape)
#     augmented_labels = np.zeros(X_train.shape[:-1])
#
#     for i in range(X_train.shape[0]):
#         # print(train_images.dtype)
#         # img = touint8(X_train[i])
#         img = X_train[i]
#         aug = transform(image=img, mask=np.argmax(Y_train[i], 2))
#         augmented_image = aug['image']
#         augmented_label = aug['mask']
#         augmented_images[i, :, :] = augmented_image
#         augmented_labels[i, :, :] = augmented_label
#
#     # augmented_labels = onehotfrompixels(augmented_labels)
#     return augmented_images, augmented_labels

def onehotfrompixels(label):
    label_ = np.zeros([label.shape[0],label.shape[1],label.shape[2],10])
    for i in range(label.shape[0]):

        for layer in range(10):
            label_[i,:,:,layer] = label[i,...]==layer

    return label_


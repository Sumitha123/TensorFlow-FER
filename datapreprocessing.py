import tensorflow as tf
import matplotlib.cm as cm
import numpy as np
import input_data
images = input_data.train_dataset()

#Subtracting the mean value of each image
images = images - images.mean(axis=1).reshape(-1,1)

#Setting the image norm to be 100
images = np.multiply(images,100.0/255.0)
pixel_mean = images.mean(axis=0)
pixel_std = np.std(images, axis=0)
images = np.divide(np.subtract(images,pixel_mean), pixel_std)

print('\nImage shape : ',images.shape)

image_pixels = images.shape[1]
print('\nFlat pixel value : ',image_pixels)

image_width = image_height = np.ceil(np.sqrt(image_pixels)).astype(np.uint8)

print('\nImage width : ', image_width)

flat_labels = input_data.training_data["emotion"].values.ravel()
labels_count = np.unique(flat_labels).shape[0]
print('Number of facial expressions : ', labels_count)
def dense_to_one_hot_encoding(dense_labels, num_emotions):
    num_labels = dense_labels.shape[0]
    index_offset = np.arange(num_labels) * num_emotions
    labels_one_hot_encoding = np.zeros((num_labels, num_emotions))
    labels_one_hot_encoding.flat[index_offset + dense_labels.ravel()] = 1
    return labels_one_hot_encoding
labels = dense_to_one_hot_encoding(flat_labels, labels_count)
labels = labels.astype(np.uint8)


#One hot encoding for emotion classes
print('\nOne hot encoding for Angry emotions : ', labels[0])


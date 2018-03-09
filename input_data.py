import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import dask.dataframe as dd

#Fetching input image dataset from fer2013.csv
image_data = pd.read_csv('/Users/sumitha/Documents/ProjectDeepLearning/capstone/fer2013.csv' )

#Checking  the number of images in the image dataset and the shape of the dataset
print('\nShape of the dataset : ',str(image_data.shape),"\n")

#View image dataset values
print(image_data.head())

#Determining the unique values in "Usage" of the image dataset
usage = np.unique(image_data["Usage"].values.ravel())
print('\nThe image dataset is split into ', str(usage))

print('\nNumber of images in dataset')
print('Training dataset   : ',str(len(image_data[image_data.Usage == "Training"])))
print('Private Test dataset    : ',str(len(image_data[image_data.Usage == "PrivateTest"])))
print('Public Test dataset : ',str(len(image_data[image_data.Usage == "PublicTest"])))

#Training dataset summary

training_data = image_data[image_data.Usage == "Training"]
print('\nTraining set summary')
print('Angry   : ',str(len(training_data[training_data.emotion == 0])))
print('Fear    : ',str(len(training_data[training_data.emotion == 1])))
print('Disgust : ',str(len(training_data[training_data.emotion == 2])))
print('Happy   : ',str(len(training_data[training_data.emotion == 3])))
print('Sad     : ',str(len(training_data[training_data.emotion == 4])))
print('Surprise: ',str(len(training_data[training_data.emotion == 5])))
print('Neutral : ',str(len(training_data[training_data.emotion == 6])))

#Function to return the images in training dataset
def train_dataset():
    training_data = image_data[image_data.Usage == "Training"]
    #Return the array as python list
    image_pixels = training_data.pixels.str.split(" ").tolist()
    image_pixels = pd.DataFrame(image_pixels, dtype=int)
    images = image_pixels.values
    images = images.astype(np.float)
    return images

#Function to show images

def show(img):
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 5
    for i in range(1, columns*rows + 1):
        show_image = img[i-1].reshape(48, 48)
        fig.add_subplot(rows, columns, i)
        plt.imshow(show_image, cmap='gray')
        plt.savefig("input_overview.png")

show(train_dataset())


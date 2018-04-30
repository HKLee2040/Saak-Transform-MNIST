#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:45:58 2018

@author: Bin Wang
"""


# arguments define
import argparse

# load torch
import torchvision

# other utilities
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#%% Load the training data
def MNIST_DATASET_TRAIN(downloads, train_amount, padding, threshold):
    # Load dataset
    training_data = torchvision.datasets.MNIST(
              root = './mnist/',
              train = True,
              transform = torchvision.transforms.ToTensor(),
              download = downloads
              )
    

    #Convert Training data to numpy
    train_data_ori = training_data.train_data.numpy()[:train_amount]
    train_label = training_data.train_labels.numpy()[:train_amount]
  
    #Convert the size to 32 by 32
    if padding == True: # Change to 32 by 32 by zero padding
        train_data = np.zeros((train_amount, 32, 32))
        for i in range(train_amount):
            train_data[i] = cv2.copyMakeBorder(train_data_ori[i],2,2,2,2,cv2.BORDER_CONSTANT, value=0)

    elif padding == False: # Change to 32 by 32 by resizing
        train_data = np.zeros((train_amount, 32, 32))
        for i in range(train_amount):
            train_data[i] = cv2.resize(train_data_ori[i],(32,32),interpolation=cv2.INTER_CUBIC)
    
    
    # Threshold the image
    if threshold != 0:
        train_data = train_data.astype(np.uint8)
        train_data = cv2.threshold(train_data, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Print training data size
    print('Training data size: ',train_data.shape)
    print('Training data label size:',train_label.shape)   
    plt.imshow(train_data[0])
    plt.show()
    
    train_data = train_data/255.0
    
    return train_data, train_label



#%% Load the test data
def MNIST_DATASET_TEST(downloads, test_amount, padding, threshold):
    # Load dataset
    testing_data = torchvision.datasets.MNIST(
              root = './mnist/',
              train = False,
              transform = torchvision.transforms.ToTensor(),
              download = downloads
              )
    
    # Convert Testing data to numpy
    test_data_ori = testing_data.test_data.numpy()[:test_amount]
    test_label = testing_data.test_labels.numpy()[:test_amount]
    
    # Convert the size to 32 by 32
    if padding == True: # Change to 32 by 32 by zero padding
        test_data = np.zeros((test_amount, 32, 32))
        for i in range(test_amount):
            test_data[i] = cv2.copyMakeBorder(test_data_ori[i],2,2,2,2,cv2.BORDER_CONSTANT, value=0)
    elif padding == False: # Change to 32 by 32 by resizing
        test_data = np.zeros((test_amount, 32, 32))
        for i in range(test_amount):
            test_data[i] = cv2.resize(test_data_ori[i],(32,32),interpolation=cv2.INTER_CUBIC)
    
    # Threshold the image
    if threshold != 0:
        test_data = test_data.astype(np.uint8)
        test_data = cv2.threshold(test_data, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Print training data size
    print('test data size: ',test_data.shape)
    print('test data label size:',test_label.shape)   
    plt.imshow(test_data[0])
    plt.show()
    
    test_data = test_data/255.0
    
    return test_data, test_label

#%% Reshape to PCA shape
def PCA_shaping(data, steps):
    #print('----------Reshaping---------- This may take some time ---')
    total_amount = int(data.shape[0] * data.shape[1]/steps * data.shape[2]/steps)
    length = steps*steps*data.shape[3]
    PCA_data = np.zeros((total_amount,length))
    
    count = 0
    for num in range(data.shape[0]):
        single_image = data[num]
        for row in range(0, data.shape[1], steps):
            for column in range(0, data.shape[2], steps):
                PCA_data[count,:] = np.reshape(single_image[row:row+steps,column:column+steps,:],-1)
                count += 1

    return PCA_data
#%% Remove low variance data
def remove_low_variance(PCA_shape_data):
    print('----------Removing low variance patches-------')
 
    variance = np.var(PCA_shape_data, axis=1)
    PCA_shape_data = PCA_shape_data[variance!=0,:]  
    
    return PCA_shape_data

#%% PCA
def conduct_PCA(PCA_shape_data, num_remaining):
    print('----------Conducting PCA-------------')
    pca = PCA(n_components=num_remaining)
    pca.fit(PCA_shape_data)
    
    return pca.components_, pca


#%% Reshape to cuboid for next stage
def reshape_cuboid(data, steps, out_size):
    num_image = int(data.shape[0]/(out_size**2))
    cuboid = np.zeros((num_image, out_size, out_size, data.shape[1]))
    count = 0
    for num in range(num_image):
        for row in range(out_size):
            for column in range(out_size):
                cuboid[num, row, column,:] = data[count,:]
                count += 1

    return cuboid

#%% Conduct one stage saak transform
def train_saak_transform(train_data, steps, num_remaining, augment, remove_low_var=False):

    out_size = int(train_data.shape[1]/steps)
    #Reshaping to fit PCA computation
    PCA_shape_data = PCA_shaping(train_data, steps)
    

    # Remove and save the DC component
    DC_feature = np.mean(PCA_shape_data,axis=1,keepdims=1)

    # For the first stage, we remove the low variance
    if remove_low_var == True:
        PCA_shape_data = remove_low_variance(PCA_shape_data)
   
    PCA_shape_data = PCA_shape_data - np.mean(PCA_shape_data,axis=1,keepdims=1)
    
    # Get the PCA kernal
    kernal,pca = conduct_PCA(PCA_shape_data,num_remaining)

    # For the first stage get back original patch
    if remove_low_var == True:
        PCA_shape_data = PCA_shaping(train_data, steps)
    
    # Conduct PCA transform
    PCA_shape_data = PCA_shape_data - DC_feature    
    data_after_PCA = pca.transform(PCA_shape_data)
    

    # Augmented kernal (SP conversion + ReLU)
    data_augmented = np.concatenate((DC_feature, data_after_PCA, (-1)*data_after_PCA),axis=1)
    data_augmented[data_augmented<0] = 0

    # Training data
    data_unaug2 = np.concatenate((DC_feature, data_after_PCA), axis = 1)
    # Reshape back to cuboid
    cuboid = reshape_cuboid(data_augmented, steps, out_size)
    if augment == 0:
         data_unaug = reshape_cuboid(data_unaug2, steps, out_size)
    elif augment == 1:
         data_unaug = data_augmented
     
     
    return cuboid, data_unaug, pca

#%% Functions for testing stage----------------------------------------------------------

def conduct_test_PCA(data, step, pca, augment):
     
    out_size = int(data.shape[1]/step)
    
    # Reshape to fit PCA computation
    PCA_shape_data = PCA_shaping(data, step)

    # Remove and save the DC component
    DC_feature = np.mean(PCA_shape_data,axis=1,keepdims=1)
    PCA_shape_data = PCA_shape_data - DC_feature
    
    # Conduct PCA transform
    data_after_PCA = pca.transform(PCA_shape_data)
    
    # Augmented kernal (SP conversion + ReLU)
    data_augmented = np.concatenate((DC_feature, data_after_PCA, (-1)*data_after_PCA),axis=1)
    data_augmented[data_augmented<0] = 0
    
    # Training data
    data_unaug2 = np.concatenate((DC_feature, data_after_PCA), axis = 1)
    
    # Reshape back to cuboid
    cuboid = reshape_cuboid(data_augmented, step, out_size)
    if augment == 0:
         data_unaug = reshape_cuboid(data_unaug2, step, out_size)
    elif augment == 1:
         data_unaug = data_augmented
         
    return cuboid, data_unaug

#%% Prepare test features for training
def test_features_extraction(data, steps, pca1, pca2,
                             pca3, pca4, pca5, pca6,F_index,augment):

 
     # Get the cuboid features for testing data
    cuboid_1,data_unaug1 = conduct_test_PCA(data, steps[0], pca1,augment)
    cuboid_2,data_unaug2 = conduct_test_PCA(cuboid_1, steps[1], pca2,augment)
    cuboid_3,data_unaug3 = conduct_test_PCA(cuboid_2, steps[2], pca3,augment)
    cuboid_4,data_unaug4 = conduct_test_PCA(cuboid_3, steps[3], pca4,augment)
    cuboid_5,data_unaug5 = conduct_test_PCA(cuboid_4, steps[4], pca5,augment)

    # Form features for each patch 2174 dimension
    data_unaug1 = np.reshape(data_unaug1, (data.shape[0],-1))
    data_unaug2 = np.reshape(data_unaug2, (data.shape[0],-1))
    data_unaug3 = np.reshape(data_unaug3, (data.shape[0],-1))
    data_unaug4 = np.reshape(data_unaug4, (data.shape[0],-1))
    data_unaug5 = np.reshape(data_unaug5, (data.shape[0],-1))
    
    test_features_long = np.concatenate((data_unaug1, data_unaug2, data_unaug3, data_unaug4, data_unaug5),axis = 1)
    
    # Select F-test features
    test_features = test_features_long[:, F_index]

    # Conduct last stage PCA
    test_features = pca6.transform(test_features)

    return test_features


#%% Main function for MNIST dataset    
if __name__=='__main__':

    # Training Arguments Settings
    parser = argparse.ArgumentParser(description='Saak')
    
    parser.add_argument('--download_MNIST', default=True, metavar='DL',
                        help='Download MNIST (default: True)')
    parser.add_argument('--train_amount', type=int, default=60000,
                        help='Amount of training samples')
    parser.add_argument('--test_amount', type=int, default=10000,
                        help='Amount of testing samples')
    parser.add_argument('--padding32', type=int, default=True,
                        help='True - padding, False - resizing') 
    parser.add_argument('--threshold_binary', type=int, default=0,
                        help='threshold to binary image, if = 0, no thresholding') 
    parser.add_argument('--steps',nargs=5,action='append', default=[2, 2, 2, 2,2],
                        help='set the window size for each stage') 
    parser.add_argument('--num_remaining', nargs=5,action='append', default=['3', '4', '7', '6', '8', '64'],
                        help='number of components to be kept') 
    parser.add_argument('--F_test_remaining',  type=int, default=1000, 
                        help='number of components to be kept for f test') 
    parser.add_argument('--svm_rf',  type=int, default=1,
                        help='choose svm or rf') 
    parser.add_argument('--augment',  type=int, default=0,
                        help='use agumented for training or not') 
    args = parser.parse_args()
    
    # Print Arguments
    print('\n----------Argument Values-----------')
    for name, value in vars(args).items():
        print('%s: %s' % (str(name), str(value)))
    print('------------------------------------\n')
    
    
    # Load Training Data & Testing Data
    train_data, train_label = MNIST_DATASET_TRAIN(args.download_MNIST, 
                                                  args.train_amount, args.padding32, args.threshold_binary)
    test_data, test_label = MNIST_DATASET_TEST(args.download_MNIST, 
                                               args.test_amount, args.padding32, args.threshold_binary)
    train_data = np.expand_dims(train_data, axis = 3)
    test_data = np.expand_dims(test_data, axis = 3)
    
    
    # Saak Transform for five stages
    print('\n\n-----------First stage saak--------------------')
    cuboid_1, data_aug1, pca1 = train_saak_transform(train_data, int(args.steps[0]),
                                                                int(args.num_remaining[0]), args.augment, remove_low_var=True)
    print('-----------Second stage saak--------------------')
    cuboid_2, data_aug2, pca2 = train_saak_transform(cuboid_1, int(args.steps[1]),
                                                                int(args.num_remaining[1]), args.augment)
    print('-----------Third stage saak--------------------')
    cuboid_3, data_aug3, pca3 = train_saak_transform(cuboid_2, int(args.steps[2]),
                                                                int(args.num_remaining[2]), args.augment)
    print('-----------Forth stage saak--------------------')
    cuboid_4, data_aug4, pca4 = train_saak_transform(cuboid_3, int(args.steps[3]),
                                                                int(args.num_remaining[3]), args.augment)
    print('-----------Fifth stage saak--------------------')
    cuboid_5, data_aug5, pca5 = train_saak_transform(cuboid_4, int(args.steps[4]),
                                                                int(args.num_remaining[4]), args.augment)
    print('-----------Finish Training Saak Kernels----------')


    # Form features for each patch xxx dimension
    data_aug1 = np.reshape(data_aug1, (args.train_amount,-1))
    data_aug2 = np.reshape(data_aug2, (args.train_amount,-1))
    data_aug3 = np.reshape(data_aug3, (args.train_amount,-1))
    data_aug4 = np.reshape(data_aug4, (args.train_amount,-1))
    data_aug5 = np.reshape(data_aug5, (args.train_amount,-1))
    
    training_feature_long = np.concatenate((data_aug1, data_aug2, data_aug3, data_aug4, data_aug5),axis = 1)
  
    #%% Conduct F test and final dimension selection
    #F-test
    F, p_value = f_classif(training_feature_long, train_label)
    F[np.isnan(F)]=0

    F_index = F > (F[np.argsort(F)][-(args.F_test_remaining+1)])
    training_features = training_feature_long[:, F_index]

    # Final stage PCA    
    _, pca6 = conduct_PCA(training_features, int(args.num_remaining[5]))
    training_features = pca6.transform(training_features)
    
    
    # Prepare test features
    test_features = test_features_extraction(test_data, args.steps, pca1, pca2,
                                             pca3, pca4, pca5, pca6,F_index,args.augment)

    
    #%% Train SVM or RF
    if args.svm_rf == 1:
         # Training SVM
         print('------Training and testing SVM------')
         clf = svm.SVC(C=5, gamma=0.05)
         #clf = svm.LinearSVC(C=5)
         clf.fit(training_features, train_label)
         
         # Test on training data
         train_result = clf.predict(training_features)
         precision = sum(train_result == train_label)/train_label.shape[0]
         print('Train precision: ', precision)
                  
         # Test on testing data
         test_result = clf.predict(test_features)
         precision = sum(test_result == test_label)/test_label.shape[0]
         print('Test precision: ', precision)
         #Show the confusion matrix
         matrix = confusion_matrix(test_label, test_result) 
         
    if args.svm_rf == 1:
         # Training RF
         print('------Training and testing RF------')
         clf = RandomForestClassifier(n_estimators=300,max_depth=40)
         clf.fit(training_features, train_label)
         
         # Test on training data
         train_result = clf.predict(training_features)
         precision = sum(train_result == train_label)/train_label.shape[0]
         print('Train precision: ', precision)
                  
         # Test on testing data
         test_result = clf.predict(test_features)
         precision = sum(test_result == test_label)/test_label.shape[0]
         print('Test precision: ', precision)
           
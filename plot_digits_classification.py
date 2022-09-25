"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""


# Standard scientific Python imports
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from itertools import product
from nltk.corpus import movie_reviews
import numpy as np
import pandas as pd
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import statistics


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
    


# flatten the images
def pre_processing(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    list_imgs=[]

    for image in dataset.images:
        #img=rescale(image, 0.5, anti_aliasing=False)
        #img= downscale_local_mean(image, (2, 3))
        img = resize(image, (image.shape[0] , image.shape[1] ),anti_aliasing=True) #no resizing here
        list_imgs.append(img)

    data = np.array(list_imgs).reshape((n_samples,-1))
    return data

def model_run(data,target,param,train_split,dev_split,test_split):
    # Create a classifier: a support vector classifier
    
    best_accuracy =0
    best_param=params[0]
    best_dev_accuracy = 0
    final_result= pd.DataFrame(columns=['params','train_accuracy','dev_accuracy','test_accuracy'])
    #print('Train , dev, test split is chaged to 0.5,0.25,0.25')
    for param in params:
        clf = svm.SVC(gamma=param[0],C=param[1])
        X_train, X_Group, y_train, y_Group = train_test_split(
            data, target, test_size=(1-train_split), shuffle=False
        )

        X_val, X_test, y_val, y_test = train_test_split(
            data, target, test_size=test_split, shuffle=False
        )

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted_train = clf.predict(X_train)
        predicted = clf.predict(X_val)
        predicted_test = clf.predict(X_test)

        ###############################################################################
        # Below we visualize the first 4 test samples and show their predicted
        # digit value in the title.

        # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        # for ax, image, prediction in zip(axes, X_test, predicted):
        #     ax.set_axis_off()
        #     image = image.reshape(8, 8)
        #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        #     ax.set_title(f"Prediction: {prediction}")

        
        
        dev_accuracy = metrics.accuracy_score(y_val, predicted)
        test_accuracy = metrics.accuracy_score(y_test, predicted_test)
        train_accuracy = metrics.accuracy_score(y_train, predicted_train)
        
        final_result.loc[len(final_result.index)] = [param, train_accuracy,dev_accuracy, test_accuracy] 
        if best_accuracy <test_accuracy:
            best_accuracy=test_accuracy
            best_param = param
            best_dev_accuracy = dev_accuracy


    final_result.index = final_result['params']
    final_result.drop(columns=['params'],inplace=True)
    print("Result")
    print(final_result)

    train_min = min(final_result['train_accuracy'])
    train_max = max(final_result['train_accuracy'])
    train_median = statistics.median(final_result['train_accuracy'])

    train_min = min(final_result['train_accuracy'])
    train_max = max(final_result['train_accuracy'])
    train_median = statistics.median(final_result['train_accuracy'])

    dev_min = min(final_result['dev_accuracy'])
    dev_max = max(final_result['dev_accuracy'])
    dev_median = statistics.median(final_result['dev_accuracy'])


    test_min = min(final_result['test_accuracy'])
    test_max = max(final_result['test_accuracy'])
    test_median = statistics.median(final_result['test_accuracy'])

    print(f"Min Accuracy for Train data  {train_min}:\n"
        f"Max  accuracy is {train_max} :\n "
        f"Median  accuracy is {train_median}\n")

    print(f"Min Accuracy for dev data  {dev_min}:\n"
        f"Max  accuracy is {dev_max} :\n "
        f"Median  accuracy is {dev_median}\n")

    print(f"Min Accuracy for Test data  {test_min}:\n"
        f"Max  accuracy is {test_max} :\n "
        f"Median  accuracy is {test_median}\n")

    print(f"Best Hyperparameters are  {best_param}:\n"
        f"Best Test accuracy is {best_accuracy}\n")


gamma_list = [0.005,0.001,0.005,0.0001,0.002]
c_list=[0.1,0.3,0.6,5,10,15,20]
params= list(product(gamma_list,c_list))

data = pre_processing(digits)
model_run(data,digits.target,params,0.6,0.2,0.2)


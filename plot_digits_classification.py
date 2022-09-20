"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

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

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
    



###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
list_imgs=[]

for image in digits.images:
    #img=rescale(image, 0.5, anti_aliasing=False)
    #img= downscale_local_mean(image, (2, 3))
    img = resize(image, (image.shape[0] // 3, image.shape[1] // 2),anti_aliasing=True)
    list_imgs.append(img)

data = np.array(list_imgs).reshape((n_samples,-1))
print('Changed Image Size')
print(list_imgs[0].shape)

# Create a classifier: a support vector classifier
gamma_list = [0.005,0.001,0.005,0.0001,0.002]
c_list=[0.1,0.3,0.6,5,10,15,20]
params = list(product(gamma_list,c_list))
best_accuracy =0
best_param=params[0]
best_dev_accuracy = 0
final_result= pd.DataFrame(columns=['params','dev_accuracy','test_accuracy'])
for param in params:
    clf = svm.SVC(gamma=param[0],C=param[1])
    #clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_Group, y_train, y_Group = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    X_val, X_test, y_val, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=False
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
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

    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.

    #print(
    #    f"Classification report for classifier {clf}:\n"
    #    f"{metrics.classification_report(y_test, predicted)}\n"
    #)

    ###############################################################################
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.

    #disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    #disp.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")
    dev_accuracy = metrics.accuracy_score(y_val, predicted)
    test_accuracy = metrics.accuracy_score(y_test, predicted_test)
    final_result.loc[len(final_result.index)] = [param, dev_accuracy, test_accuracy] 
    if best_accuracy <test_accuracy:
        best_accuracy=test_accuracy
        best_param = param
        best_dev_accuracy = dev_accuracy

final_result.index = final_result['params']
final_result.drop(columns=['params'],inplace=True)
print("Result")
print(final_result)

print(f"Best Hyperparameters are  {best_param}:\n"
    f"Best Dev accuracy is {best_dev_accuracy} :\n "
    f"Best Test accuracy is {best_accuracy}\n")
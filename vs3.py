"""
    Name: John Huynh
    GT ID: XX3956035
    Dataset: Fruits - k-nearest neighbors
    Assignment: 1
    Citation: Mihai Oltean, Fruits-360 dataset, 2017-
    https://www.kaggle.com/moltean/fruits
    https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
"""

from PIL import Image
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import NuSVC, SVC
import skimage.io as io
import skimage as ski
import cv2
from matplotlib.image import imread
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

def load_images_from_folders(base_path):
    image_data = []
    labels = []

    for folder_name in os.listdir(base_path):
        #folder_name = "/" + folder_name
        folder_path = os.path.join(base_path, folder_name)
        #print(folder_path)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                backslash1 = image_path.find('\\')
                first = image_path[:backslash1]
                backslash2 = image_path.find('\\',(backslash1 + 1))
                second = image_path[(backslash1 + 1):backslash2]
                third = image_path[(backslash2 + 1):]
                image_path = first + "/" + second + "/" + third
                if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
                    image = cv2.imread(image_path)
                    image = resize(
                        image, (25, 25), anti_aliasing=True
                    )

                    image = np.asarray(image, dtype=object)  # Convert the image to a NumPy array
                    image = image / 255.0

                    image = image.flatten()
                    #print("image = ", image)
                    image_data.append(image)
                    labels.append(folder_name)  # Store the folder name as the label

    return np.array(image_data), labels

def load_images_from_folders_test(base_path):
    image_data = []
    labels = []

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        #print(folder_path)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                backslash1 = image_path.find('\\')
                first = image_path[:backslash1]
                backslash2 = image_path.find('\\',(backslash1 + 1))
                second = image_path[(backslash1 + 1):backslash2]
                third = image_path[(backslash2 + 1):]
                image_path = first + "/" + second + "/" + third
                #print(image_path)
                if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
                    #image = ski.io.imread(image_path)
                    #image = Image.open(image_path)  # Open image using PIL
                    image = cv2.imread(image_path)

                    #image = rescale(image, 0.25, anti_aliasing=False)
                    image = resize(
                        image, (25, 25), anti_aliasing=True
                    )

                    image = np.array(image)  # Convert the image to a NumPy array
                    image = image / 255.0

                    image = image.flatten()
                    image_data.append(image)
                    labels.append(folder_name)  # Store the folder name as the label

    return np.array(image_data), labels

def visualize(predicted, X_test, y_test, clf):

    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        print("Plot_image.shape = ", image.shape)
        image = image.reshape(25, 25, 3)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    ###############################################################################
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

    ###############################################################################
    # If the results from evaluating a classifier are stored in the form of a
    # :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
    # `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
    # as follows:


    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )

def k_nearest_neighbors(k, X_train, y_train, X_test, y_test):
    # Nearest neighbors
    clf = Pipeline(steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])

    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    print(predicted)

    print("Accuracy score with normalization = ", accuracy_score(y_test, predicted))

    print("Accuracy score without normalization = ", accuracy_score(y_test, predicted, normalize=False))

    print("Training set score: %f" % clf.score(X_train, y_train))
    print("Test set score: %f" % clf.score(X_test, y_test))
    visualize(predicted, X_test, y_test, clf)

base_directory = "./Fruits/fruits-360_dataset_original-size/fruits-360-original-size/Training"
X_train, y_train = load_images_from_folders(base_directory)

base_directory_test = "./Fruits/fruits-360_dataset_original-size/fruits-360-original-size/Test"
X_test, y_test = load_images_from_folders_test(base_directory_test)
print("X_test.shape = ", X_test.shape)
print("X_test[0] = ", X_test[0])

print(X_train[0])

# k = 10
k_nearest_neighbors(10, X_train, y_train, X_test, y_test)

# k = 15
k_nearest_neighbors(15, X_train, y_train, X_test, y_test)

"""
# KNN graphs
_, axs = plt.subplots(ncols=2, figsize=(12, 5))

for ax, weights in zip(axs, ("uniform", "distance")):
    clf.set_params(knn__weights=weights).fit(images, labels)
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_test,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel="x",
        ylabel="y",
        shading="auto",
        alpha=0.5,
        ax=ax,
    )
    scatter = disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors="k")
    disp.ax_.legend(
        scatter.legend_elements()[0],
        y_test,
        loc="lower left",
        title="Classes",
    )
    _ = disp.ax_.set_title(
        f"3-Class classification\n(k={clf[-1].n_neighbors}, weights={weights!r})"
    )

plt.show()
"""

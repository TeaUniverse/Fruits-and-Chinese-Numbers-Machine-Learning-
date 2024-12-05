"""
    Name: John Huynh
    GT ID: XX3956035
    Dataset: Chinese numbers - support vector machines
    Assignment: 1
    Citation: Nazarpour, K; Chen, M (2017). Handwritten Chinese Numbers. Newcastle University. Dataset. https://doi.org/10.17634/137930-3
    https://doi.org/10.17634/137930-3
"""

from PIL import Image
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import NuSVC, SVC, NuSVR
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

def load_images_from_folders(base_path):
    image_data = []
    labels = []
    #print(len(os.listdir(base_path)))
    for repetition in range(len(os.listdir(base_path))//15):
        for i in range(11): # represents 1-10
            labels.append(i + 1)
        labels.append(100) # represents hundred(s)
        labels.append(1000) # represents thousand(s)
        labels.append(10000) # represents ten(s) of thousand
        labels.append(100000000) # represents hundred(s) of million

    for image_name in os.listdir(base_path):
        image_path = os.path.join(base_path, image_name)
        #print("image_path = ", image_path)
        if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
            #image = ski.io.imread(image_path)
            #image = Image.open(image_path)  # Open image using PIL
            image = cv2.imread(image_path)
            #print("image.shape = ", image.shape)
            #image = rescale(image, 0.25, anti_aliasing=False)
            image = resize(
                image, (16, 16), anti_aliasing=True
            )
            #print("image.shape = ", image.shape)

            image = np.asarray(image) # Convert the image to a NumPy array
            #image = np.asarray(image, dtype=object)  # Convert the image to a NumPy array
            image = image / 255.0

            image = image.flatten()
            #print("image = ", image)
            image_data.append(image)

    return np.array(image_data), labels

def visualize(predicted, X_test, y_test, clf):

    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        print("Plot_image.shape = ", image.shape)
        image = image.reshape(16, 16, 3)
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

def svm2(kernel, X_train, X_test, y_train, y_test):
    # Support vector machines
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)

    print("X_test.shape = ", X_test.shape)
    print("X_test[0] = ", X_test[0])

    predicted = clf.predict(X_test)
    print(predicted)

    print("Accuracy score with normalization = ", accuracy_score(y_test, predicted))

    print("Accuracy score without normalization = ", accuracy_score(y_test, predicted, normalize=False))

    print("Training set score: %f" % clf.score(X_train, y_train))
    print("Test set score: %f" % clf.score(X_test, y_test))
    visualize(predicted, X_test, y_test, clf)

base_directory = "./ChineseNumbers/data/data"
images, labels = load_images_from_folders(base_directory)

X_train, X_test, y_train, y_test = train_test_split(images, labels, stratify=labels, random_state=0)

svm1_data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
# clf = make_pipeline(StandardScaler(), NuSVC())

print(images[0])

# kernel = 'rbf'
#svm2('rbf', X_train, X_test, y_train, y_test)

# kernel = 'poly'
#svm2('poly', X_train, X_test, y_train, y_test)

"""
=========================================================
Plotting Learning Curves and Checking Models' Scalability
=========================================================

In this example, we show how to use the class
:class:`~sklearn.model_selection.LearningCurveDisplay` to easily plot learning
curves. In addition, we give an interpretation to the learning curves obtained
for a naive Bayes and SVM classifiers.

Then, we explore and draw some conclusions about the scalability of these predictive
models by looking at their computational cost and not only at their statistical
accuracy.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Learning Curve
# ==============
#
# Learning curves show the effect of adding more samples during the training
# process. The effect is depicted by checking the statistical performance of
# the model in terms of training score and testing score.
#
# Here, we compute the learning curve of a naive Bayes classifier and a SVM
# classifier with a RBF kernel using the digits dataset.

#print("X_train = ", X_train)
#print("X_test = ", X_test)


"""
X = images
y = labels
#svc_rbf = SVC(kernel="rbf", gamma=0.001)
#svc_poly = SVC(kernel="poly", gamma=0.001)
svc_rbf = SVC(kernel="rbf")
svc_poly = SVC(kernel="poly")

# %%
# The :meth:`~sklearn.model_selection.LearningCurveDisplay.from_estimator`
# displays the learning curve given the dataset and the predictive model to
# analyze. To get an estimate of the scores uncertainty, this method uses
# a cross-validation procedure.

from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

common_params = {
    "X": X,
    "y": y,
    "train_sizes": [X.size//5, X.size//3, X.size],
    "cv": 5,    
    #"train_sizes": np.linspace(0.1, 1.0, 5),
    #"cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}

for ax_idx, estimator in enumerate([svc_rbf, svc_poly]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")

# %%
# We first analyze the learning curve of the naive Bayes classifier. Its shape
# can be found in more complex datasets very often: the training score is very
# high when using few samples for training and decreases when increasing the
# number of samples, whereas the test score is very low at the beginning and
# then increases when adding samples. The training and test scores become more
# realistic when all the samples are used for training.
#
# We see another typical learning curve for the SVM classifier with RBF kernel.
# The training score remains high regardless of the size of the training set.
# On the other hand, the test score increases with the size of the training
# dataset. Indeed, it increases up to a point where it reaches a plateau.
# Observing such a plateau is an indication that it might not be useful to
# acquire new data to train the model since the generalization performance of
# the model will not increase anymore.
#
# Complexity analysis
# ===================
#
# In addition to these learning curves, it is also possible to look at the
# scalability of the predictive models in terms of training and scoring times.
#
# The :class:`~sklearn.model_selection.LearningCurveDisplay` class does not
# provide such information. We need to resort to the
# :func:`~sklearn.model_selection.learning_curve` function instead and make
# the plot manually.

# %%
from sklearn.model_selection import learning_curve

common_params = {
    "X": X,
    "y": y,
    "train_sizes": [X.size//5, X.size//3, X.size],
    "cv": 5,
    #"train_sizes": np.linspace(0.1, 1.0, 5),
    #"cv": ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
    "n_jobs": 4,
    "return_times": True,
}

train_sizes, _, test_scores_rbf, fit_times_rbf, score_times_rbf = learning_curve(
    svc_rbf, **common_params
)
train_sizes, _, test_scores_poly, fit_times_poly, score_times_poly = learning_curve(
    svc_poly, **common_params
)

# %%
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=True)

for ax_idx, (fit_times, score_times, estimator) in enumerate(
    zip(
        [fit_times_rbf, fit_times_poly],
        [score_times_rbf, score_times_poly],
        [svc_rbf, svc_poly],
    )
):
    # scalability regarding the fit time
    ax[0, ax_idx].plot(train_sizes, fit_times.mean(axis=1), "o-")
    ax[0, ax_idx].fill_between(
        train_sizes,
        fit_times.mean(axis=1) - fit_times.std(axis=1),
        fit_times.mean(axis=1) + fit_times.std(axis=1),
        alpha=0.3,
    )
    ax[0, ax_idx].set_ylabel("Fit time (s)")
    ax[0, ax_idx].set_title(
        f"Scalability of the {estimator.__class__.__name__} classifier"
    )

    # scalability regarding the score time
    ax[1, ax_idx].plot(train_sizes, score_times.mean(axis=1), "o-")
    ax[1, ax_idx].fill_between(
        train_sizes,
        score_times.mean(axis=1) - score_times.std(axis=1),
        score_times.mean(axis=1) + score_times.std(axis=1),
        alpha=0.3,
    )
    ax[1, ax_idx].set_ylabel("Score time (s)")
    ax[1, ax_idx].set_xlabel("Number of training samples")

# %%
# We see that the scalability of the SVM and naive Bayes classifiers is very
# different. The SVM classifier complexity at fit and score time increases
# rapidly with the number of samples. Indeed, it is known that the fit time
# complexity of this classifier is more than quadratic with the number of
# samples which makes it hard to scale to dataset with more than a few
# 10,000 samples. In contrast, the naive Bayes classifier scales much better
# with a lower complexity at fit and score time.
#
# Subsequently, we can check the trade-off between increased training time and
# the cross-validation score.

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

for ax_idx, (fit_times, test_scores, estimator) in enumerate(
    zip(
        [fit_times_rbf, fit_times_poly],
        [test_scores_rbf, test_scores_poly],
        [svc_rbf, svc_poly],
    )
):
    ax[ax_idx].plot(fit_times.mean(axis=1), test_scores.mean(axis=1), "o-")
    ax[ax_idx].fill_between(
        fit_times.mean(axis=1),
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        alpha=0.3,
    )
    ax[ax_idx].set_ylabel("Accuracy")
    ax[ax_idx].set_xlabel("Fit time (s)")
    ax[ax_idx].set_title(
        f"Performance of the {estimator.__class__.__name__} classifier"
    )

plt.show()

# %%
# In these plots, we can look for the inflection point for which the
# cross-validation score does not increase anymore and only the training time
# increases.
"""

"""
==========================
Model Complexity Influence
==========================

Demonstrate how model complexity influences both prediction accuracy and
computational performance.

We will be using two datasets:
    - :ref:`diabetes_dataset` for regression.
      This dataset consists of 10 measurements taken from diabetes patients.
      The task is to predict disease progression;
    - :ref:`20newsgroups_dataset` for classification. This dataset consists of
      newsgroup posts. The task is to predict on which topic (out of 20 topics)
      the post is written about.

We will model the complexity influence on three different estimators:
    - :class:`~sklearn.linear_model.SGDClassifier` (for classification data)
      which implements stochastic gradient descent learning;

    - :class:`~sklearn.svm.NuSVR` (for regression data) which implements
      Nu support vector regression;

    - :class:`~sklearn.ensemble.GradientBoostingRegressor` builds an additive
      model in a forward stage-wise fashion. Notice that
      :class:`~sklearn.ensemble.HistGradientBoostingRegressor` is much faster
      than :class:`~sklearn.ensemble.GradientBoostingRegressor` starting with
      intermediate datasets (`n_samples >= 10_000`), which is not the case for
      this example.


We make the model complexity vary through the choice of relevant model
parameters in each of our selected models. Next, we will measure the influence
on both computational performance (latency) and predictive power (MSE or
Hamming Loss).

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time

from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hamming_loss, mean_squared_error

# Initialize random generator
np.random.seed(0)

##############################################################################
# Load the data
# -------------
#
# First we load both datasets.
#
# .. note:: We are using
#    :func:`~sklearn.datasets.fetch_20newsgroups_vectorized` to download 20
#    newsgroups dataset. It returns ready-to-use features.
#
# .. note:: ``X`` of the 20 newsgroups dataset is a sparse matrix while ``X``
#    of diabetes dataset is a numpy array.
#


classification_data = svm1_data


##############################################################################
# Benchmark influence
# -------------------
# Next, we can calculate the influence of the parameters on the given
# estimator. In each round, we will set the estimator with the new value of
# ``changing_param`` and we will be collecting the prediction times, prediction
# performance and complexities to see how those changes affect the estimator.
# We will calculate the complexity using ``complexity_computer`` passed as a
# parameter.
#


def benchmark_influence(conf):
    
    #Benchmark influence of `changing_param` on both MSE and latency.
    
    prediction_times = []
    prediction_powers = []
    complexities = []
    for param_value in conf["changing_param_values"]:
        conf["tuned_params"][conf["changing_param"]] = param_value
        estimator = conf["estimator"](**conf["tuned_params"])

        print("Benchmarking %s" % estimator)
        estimator.fit(conf["data"]["X_train"], conf["data"]["y_train"])
        conf["postfit_hook"](estimator)
        complexity = conf["complexity_computer"](estimator)
        complexities.append(complexity)
        start_time = time.time()
        for _ in range(conf["n_samples"]):
            y_pred = estimator.predict(conf["data"]["X_test"])
        elapsed_time = (time.time() - start_time) / float(conf["n_samples"])
        prediction_times.append(elapsed_time)
        pred_score = conf["prediction_performance_computer"](
            conf["data"]["y_test"], y_pred
        )
        prediction_powers.append(pred_score)
        print(
            "Complexity: %d | %s: %.4f | Pred. Time: %fs\n"
            % (
                complexity,
                conf["prediction_performance_label"],
                pred_score,
                elapsed_time,
            )
        )
    return prediction_powers, prediction_times, complexities


##############################################################################
# Choose parameters
# -----------------
#
# We choose the parameters for each of our estimators by making
# a dictionary with all the necessary values.
# ``changing_param`` is the name of the parameter which will vary in each
# estimator.
# Complexity will be defined by the ``complexity_label`` and calculated using
# `complexity_computer`.
# Also note that depending on the estimator type we are passing
# different data.
#


def _count_nonzero_coefficients(estimator):
    a = estimator.coef_.toarray()
    return np.count_nonzero(a)


configurations = [
    {
        "estimator": SVC,
        "tuned_params": {"kernel": 'rbf', "gamma": 2**-15},
        "changing_param": "C",
        "changing_param_values": np.logspace(-2.3, -1.3, 10),
        "complexity_label": "n_support_vectors",
        "complexity_computer": lambda x: len(x.support_vectors_),
        "data": classification_data,
        "postfit_hook": lambda x: x,
        "prediction_performance_computer": mean_squared_error,
        "prediction_performance_label": "MSE",
        "n_samples": 15,
    },
{
        "estimator": SVC,
        "tuned_params": {"kernel": 'rbf', "gamma": 2**-15},
        "changing_param": "C",
        "changing_param_values": np.logspace(-2.3, -1.3, 10),
        "complexity_label": "n_support_vectors",
        "complexity_computer": lambda x: len(x.support_vectors_),
        "data": classification_data,
        "postfit_hook": lambda x: x,
        "prediction_performance_computer": mean_squared_error,
        "prediction_performance_label": "MSE",
        "n_samples": 15,
    },
]


##############################################################################
# Run the code and plot the results
# ---------------------------------
#
# We defined all the functions required to run our benchmark. Now, we will loop
# over the different configurations that we defined previously. Subsequently,
# we can analyze the plots obtained from the benchmark:
# Relaxing the `L1` penalty in the SGD classifier reduces the prediction error
# but leads to an increase in the training time.
# We can draw a similar analysis regarding the training time which increases
# with the number of support vectors with a Nu-SVR. However, we observed that
# there is an optimal number of support vectors which reduces the prediction
# error. Indeed, too few support vectors lead to an under-fitted model while
# too many support vectors lead to an over-fitted model.
# The exact same conclusion can be drawn for the gradient-boosting model. The
# only the difference with the Nu-SVR is that having too many trees in the
# ensemble is not as detrimental.
#


def plot_influence(conf, mse_values, prediction_times, complexities):
    
    #Plot influence of model complexity on both accuracy and latency.
    

    fig = plt.figure()
    fig.subplots_adjust(right=0.75)

    # first axes (prediction error)
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(complexities, mse_values, c="tab:blue", ls="-")[0]
    ax1.set_xlabel("Model Complexity (%s)" % conf["complexity_label"])
    y1_label = conf["prediction_performance_label"]
    ax1.set_ylabel(y1_label)

    ax1.spines["left"].set_color(line1.get_color())
    ax1.yaxis.label.set_color(line1.get_color())
    ax1.tick_params(axis="y", colors=line1.get_color())

    # second axes (latency)
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    line2 = ax2.plot(complexities, prediction_times, c="tab:orange", ls="-")[0]
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    y2_label = "Time (s)"
    ax2.set_ylabel(y2_label)
    ax1.spines["right"].set_color(line2.get_color())
    ax2.yaxis.label.set_color(line2.get_color())
    ax2.tick_params(axis="y", colors=line2.get_color())

    plt.legend(
        (line1, line2), ("prediction error", "prediction latency"), loc="upper center"
    )

    plt.title(
        "Influence of varying '%s' on %s"
        % (conf["changing_param"], conf["estimator"].__name__)
    )


for conf in configurations:
    prediction_performances, prediction_times, complexities = benchmark_influence(conf)
    plot_influence(conf, prediction_performances, prediction_times, complexities)
plt.show()

##############################################################################
# Conclusion
# ----------
#
# As a conclusion, we can deduce the following insights:
#
# * a model which is more complex (or expressive) will require a larger
#   training time;
# * a more complex model does not guarantee to reduce the prediction error.
#
# These aspects are related to model generalization and avoiding model
# under-fitting or over-fitting.

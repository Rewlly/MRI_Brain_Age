"""
============================
Feature Selection
============================

An example showing univariate feature selection.

Noisy (non-informative) features are added to the iris data and
univariate feature selection is applied. For each feature, we plot the
p-values for the univariate feature selection and the corresponding
weights of an SVM. We can see that univariate feature selection
selects the informative features and that these have larger SVM weights.

In the total set of features, only the 4 first ones are significant. We
can see that they have the highest score with univariate feature
selection. The SVM assigns a large weight to one of these features, but also
Selects many of the non-informative features.
Applying univariate feature selection before the SVM
increases the SVM weight attributed to the significant features, and will
thus improve classification.

"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn.feature_selection import SelectKBest, f_classif

import BAE

# #############################################################################
# Import some data to play with

# The iris dataset
MRI_source = genfromtxt('volume_sum_icv_site.csv', delimiter=',', encoding='utf8', dtype=float)
X_train, X_test, y_train, y_test = BAE.read_source()

plt.figure(1)
plt.clf()

X_indices = np.arange(X_train.shape[-1])

# #############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function to select the four
# most significant features
selector = SelectKBest(f_classif, k=10)
selector.fit(X_train, y_train)
# scores = -np.log10(selector.pvalues_)
scores = selector.scores_
scores /= scores.max()
plt.bar(
    X_indices, scores, width=0.2, label=r"Univariate score ($-Log(p_{value})$)"
)

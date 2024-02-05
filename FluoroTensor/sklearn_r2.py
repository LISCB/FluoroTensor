# R squared calculator from sklearn.metrics This has been isolated from sklearn so the App using it can be successfully compiled without the full sklearn package
# due to hidden import errors
# Modified for simple linear regression

# BSD 3-Clause License

# Copyright (c) 2007-2023 The scikit-learn developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np


def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):

    # :math:`R^2` (coefficient of determination) regression score function.
    #
    # Best possible score is 1.0 and it can be negative (because the
    # model can be arbitrarily worse). A constant model that always
    # predicts the expected value of y, disregarding the input features,
    # would get a :math:`R^2` score of 0.0.
    #
    # Read more in the :ref:`User Guide <r2_score>`.
    #
    # Parameters
    # ----------
    # y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
    #     Ground truth (correct) target values.
    #
    # y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
    #     Estimated target values.
    #
    # sample_weight : array-like of shape (n_samples,), default=None
    #     Sample weights.
    #
    # multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}, \
    #         array-like of shape (n_outputs,) or None, default='uniform_average'
    #
    #     Defines aggregating of multiple output scores.
    #     Array-like value defines weights used to average scores.
    #     Default is "uniform_average".
    #
    #     'raw_values' :
    #         Returns a full set of scores in case of multioutput input.
    #
    #     'uniform_average' :
    #         Scores of all outputs are averaged with uniform weight.
    #
    #     'variance_weighted' :
    #         Scores of all outputs are averaged, weighted by the variances
    #         of each individual output.
    #
    #     .. versionchanged:: 0.19
    #         Default value of multioutput is 'uniform_average'.
    #
    # Returns
    # -------
    # z : float or ndarray of floats
    #     The :math:`R^2` score or ndarray of scores if 'multioutput' is
    #     'raw_values'.
    #
    # Notes
    # -----
    # This is not a symmetric function.
    #
    # Unlike most other scores, :math:`R^2` score may be negative (it need not
    # actually be the square of a quantity R).
    #
    # This metric is not well-defined for single samples and will return a NaN
    # value if n_samples is less than two.
    #
    # References
    # ----------
    # .. [1] `Wikipedia entry on the Coefficient of determination
    #         <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    #
    # Examples
    # --------
    # >>> from sklearn.metrics import r2_score
    # >>> y_true = [3, -0.5, 2, 7]
    # >>> y_pred = [2.5, 0.0, 2, 8]
    # >>> r2_score(y_true, y_pred)
    # 0.948...
    # >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    # >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    # >>> r2_score(y_true, y_pred,
    # ...          multioutput='variance_weighted')
    # 0.938...
    # >>> y_true = [1, 2, 3]
    # >>> y_pred = [1, 2, 3]
    # >>> r2_score(y_true, y_pred)
    # 1.0
    # >>> y_true = [1, 2, 3]
    # >>> y_pred = [2, 2, 2]
    # >>> r2_score(y_true, y_pred)
    # 0.0
    # >>> y_true = [1, 2, 3]
    # >>> y_pred = [3, 2, 1]
    # >>> r2_score(y_true, y_pred)
    # -3.0

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    weight = 1.0

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (
        weight * (y_true - np.average(y_true, axis=0, weights=sample_weight)) ** 2
    ).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[0]])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # return scores individually
            return output_scores
        elif multioutput == "uniform_average":
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == "variance_weighted":
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)

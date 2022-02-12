import numpy as np

from fairness.metrics.Metric import Metric


class UtilityMetric(Metric):

    def __init__(self, welfare_fn, cost_fn):
        """
        Parameters
        ----------
        welfare_fn : function
            Welfare function: (nonclass_vals, class_vals, predicted_vals, prot_index, unprot_index ) -> float
        cost_fn : function
            Cost function: (nonclass_vals, class_vals, predicted_vals) -> float
        """
        Metric.__init__(self)
        self.welfare_fn = welfare_fn
        self.cost_fn    = cost_fn

    def calc_welfare(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
            unprotected_vals, positive_pred, dict_of_nonclass_attrs):

        actual_binary = np.array(actual) == positive_pred
        # print(f'predicted: {predicted}')
        # print(f'positive_pred: {positive_pred}')
        predicted_binary = np.array(predicted) == positive_pred

        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        nonclass_attr_vals = np.array(dict_of_nonclass_attrs[single_sensitive_name])

        # Get the index of all unprotected (majority) individuals
        unprot_index = [x in unprotected_vals for x in sensitive]
        # Get the index of all protected (minority) individuals
        prot_index = [False if x else True for x in unprot_index]

        # Compute welfare for each sensitive class
        prot_welf, unprot_welf = self.welfare_fn(nonclass_attr_vals,
                                                 actual_binary,
                                                 predicted_binary,
                                                 prot_index,
                                                 unprot_index)

        return prot_welf, unprot_welf

    def calc_cost(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
            unprotected_vals, positive_pred, dict_of_nonclass_attrs):

        actual_binary = np.array(actual) == positive_pred
        # print(f'predicted: {predicted}')
        # print(f'positive_pred: {positive_pred}')
        predicted_binary = np.array(predicted) == positive_pred

        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        nonclass_attr_vals = np.array(dict_of_nonclass_attrs[single_sensitive_name])

        # Get the index of all unprotected (majority) individuals
        unprot_index = [x in unprotected_vals for x in sensitive]
        # Get the index of all protected (minority) individuals
        prot_index = [False if x else True for x in unprot_index]

        # Compute welfare for each sensitive class
        prot_cost, unprot_cost = self.cost_fn(nonclass_attr_vals,
                                              actual_binary,
                                              predicted_binary,
                                              prot_index,
                                              unprot_index)

        return prot_cost, unprot_cost


"""
The following are frequently used welfare and cost functions.
"""

def pos_rate(nonclass_attr_vals, actual_binary, predicted_binary, prot_index, unprot_index):
    """Returns the positive rate for each input row.

    Parameters
    ----------
    nonclass_attr_vals : 2D array-like
        Input attributes (X). Does not include sensitive attributes.
    actual_binary : 1D array-like<binary>
        Targets.
    predicted_binary : 1D array-like<binary>, shape==actuals
        Predicted targets.
    prot_index : 1D array-like<binary>, shape==actuals
        Index of the protected (minority) group.
    unprot_index : 1D array-like<binary>, shape==actuals
        Index of the unprotected (majorit) group.

    Returns
    -------
    Tuple <numpy.array<float>, numpy.array<float>>
        The welfare values for the protected group and unprotected group,
        respectively.
    """
    # Get the targets for each sensitive group
    prot_actual_binary   = actual_binary[prot_index]
    unprot_actual_binary = actual_binary[unprot_index]

    # Get the non-target features for each sensitive group
    prot_nonclass_vals   = nonclass_attr_vals[prot_index]
    unprot_nonclass_vals = nonclass_attr_vals[unprot_index]

    # Get the predicted values for each sensitive group
    prot_predicted_binary   = predicted_binary[prot_index]
    unprot_predicted_binary = predicted_binary[unprot_index]

    prot_welf   = prot_predicted_binary
    unprot_welf = unprot_predicted_binary

    return prot_welf, unprot_welf


def accuracy(nonclass_attr_vals, actual_binary, predicted_binary, prot_index, unprot_index):
    """Returns the accuracy for each input row.

    Parameters
    ----------
    nonclass_attr_vals : 2D array-like
        Input attributes (X). Does not include sensitive attributes.
    actual_binary : 1D array-like<binary>
        Targets.
    predicted_binary : 1D array-like<binary>, shape==actuals
        Predicted targets.
    prot_index : 1D array-like<binary>, shape==actuals
        Index of the protected (minority) group.
    unprot_index : 1D array-like<binary>, shape==actuals
        Index of the unprotected (majorit) group.

    Returns
    -------
    Tuple <numpy.array<float>, numpy.array<float>>
        The welfare values for the protected group and unprotected group,
        respectively.
    """
    # Get the targets for each sensitive group
    prot_actual_binary   = actual_binary[prot_index]
    unprot_actual_binary = actual_binary[unprot_index]

    # Get the non-target features for each sensitive group
    prot_nonclass_vals   = nonclass_attr_vals[prot_index]
    unprot_nonclass_vals = nonclass_attr_vals[unprot_index]

    # Get the predicted values for each sensitive group
    prot_predicted_binary   = predicted_binary[prot_index]
    unprot_predicted_binary = predicted_binary[unprot_index]

    prot_welf   = np.array(prot_predicted_binary == prot_actual_binary, dtype=int)
    unprot_welf = np.array(unprot_predicted_binary == unprot_actual_binary, dtype=int)

    return prot_welf, unprot_welf


def german_credit_payoffs(nonclass_attr_vals, actual_binary, predicted_binary, prot_index,
        unprot_index):
    """Returns the payoff value specified by the German Credit dataset. We also  multiply the
    payoff by the loan amount.

    ```
    This dataset requires use of a cost matrix (see below)

         1  2
       ------
    1  | 0  1
    2  | 5  0

    (1 = Good, 2 = Bad)
    The rows represent the actual classification and the columns the predicted classification.
    It is worse to class a customer as good when they are bad (5), than it is to class a customer as
    bad when they are good (1).
    ```

    Parameters
    ----------
    nonclass_attr_vals : 2D array-like
        Input attributes (X). Does not include sensitive attributes.
    actual_binary : 1D array-like<binary>
        Targets.
    predicted_binary : 1D array-like<binary>, shape==actuals
        Predicted targets.
    prot_index : 1D array-like<binary>, shape==actuals
        Index of the protected (minority) group.
    unprot_index : 1D array-like<binary>, shape==actuals
        Index of the unprotected (majorit) group.

    Returns
    -------
    Tuple <numpy.array<float>, numpy.array<float>>
        The welfare values for the protected group and unprotected group,
        respectively.
    """
    # Get the targets for each sensitive group
    prot_actual_binary   = actual_binary[prot_index]
    unprot_actual_binary = actual_binary[unprot_index]

    # Get the non-target features for each sensitive group
    prot_nonclass_vals   = nonclass_attr_vals[prot_index]
    unprot_nonclass_vals = nonclass_attr_vals[unprot_index]

    # Get the predicted values for each sensitive group
    prot_predicted_binary   = predicted_binary[prot_index]
    unprot_predicted_binary = predicted_binary[unprot_index]

    prot_welf   = np.array(prot_predicted_binary == prot_actual_binary, dtype=int)
    unprot_welf = np.array(unprot_predicted_binary == unprot_actual_binary, dtype=int)

    # TODO JDB 12/05/2021 - multiply the payoff values by the loan amount (x attribute).
    # Note that this may be different for each preprocessed dataset.

    # print(f'prot_nonclass_vals[0:10]: {prot_nonclass_vals[0:10]}')

    # Compute costs for protected group
    prot_costs = np.zeros(len(prot_actual_binary))

    for i in range(len(prot_actual_binary)):
        if prot_actual_binary[i] == prot_predicted_binary[i]:
            # Bad credit applicant was rejected OR good credit applicant was granted a loan
            prot_costs[i] = 0
        elif prot_predicted_binary[i] == 0 and prot_actual_binary[i] == 1:
            # Good credit applicant was rejected
            prot_costs[i] = 1
        else:
            # Bad credit applicant was granted a loan
            prot_costs[i] = 5

    # Compute costs for protected group
    unprot_costs = np.zeros(len(unprot_actual_binary))

    for i in range(len(unprot_actual_binary)):
        if unprot_actual_binary[i] == unprot_predicted_binary[i]:
            # Bad credit applicant was rejected OR good credit applicant was granted a loan
            unprot_costs[i] = 0
        elif unprot_predicted_binary[i] == 0 and unprot_actual_binary[i] == 1:
            # Good credit applicant was rejected
            unprot_costs[i] = 1
        else:
            # Bad credit applicant was granted a loan
            unprot_costs[i] = 5

    return prot_costs, unprot_costs


def german_credit_payoffs_imbal(nonclass_attr_vals, actual_binary, predicted_binary, prot_index,
        unprot_index):
    # Get the targets for each sensitive group
    prot_actual_binary   = actual_binary[prot_index]
    unprot_actual_binary = actual_binary[unprot_index]

    # Get the non-target features for each sensitive group
    prot_nonclass_vals   = nonclass_attr_vals[prot_index]
    unprot_nonclass_vals = nonclass_attr_vals[unprot_index]

    # Get the predicted values for each sensitive group
    prot_predicted_binary   = predicted_binary[prot_index]
    unprot_predicted_binary = predicted_binary[unprot_index]

    prot_welf   = np.array(prot_predicted_binary == prot_actual_binary, dtype=int)
    unprot_welf = np.array(unprot_predicted_binary == unprot_actual_binary, dtype=int)

    # TODO JDB 12/05/2021 - multiply the payoff values by the loan amount (x attribute).
    # Note that this may be different for each preprocessed dataset.

    # print(f'prot_nonclass_vals[0:10]: {prot_nonclass_vals[0:10]}')

    # Compute costs for protected group
    prot_costs = np.zeros(len(prot_actual_binary))

    for i in range(len(prot_actual_binary)):
        if prot_actual_binary[i] == prot_predicted_binary[i]:
            # Bad credit applicant was rejected OR good credit applicant was granted a loan
            prot_costs[i] = 0
        elif prot_predicted_binary[i] == 0 and prot_actual_binary[i] == 1:
            # Good credit applicant was rejected
            prot_costs[i] = 1
        else:
            # Bad credit applicant was granted a loan
            prot_costs[i] = 20

    # Compute costs for protected group
    unprot_costs = np.zeros(len(unprot_actual_binary))

    for i in range(len(unprot_actual_binary)):
        if unprot_actual_binary[i] == unprot_predicted_binary[i]:
            # Bad credit applicant was rejected OR good credit applicant was granted a loan
            unprot_costs[i] = 0
        elif unprot_predicted_binary[i] == 0 and unprot_actual_binary[i] == 1:
            # Good credit applicant was rejected
            unprot_costs[i] = 1
        else:
            # Bad credit applicant was granted a loan
            unprot_costs[i] = 20

    return prot_costs, unprot_costs

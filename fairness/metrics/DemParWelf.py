import sys
import numpy as np
import math

from fairness.metrics.UtilityMetric import UtilityMetric

class DemParWelf(UtilityMetric):

    def __init__(self, welfare_fn, cost_fn, tau=None, transform_welf=None, name=None):
        UtilityMetric.__init__(self, welfare_fn, cost_fn)
        self.tau = tau
        self.transform_welf = transform_welf
        if name is not None:
            self.name = f'DemParWelf_{name}'
        else:
            self.name = f'DemParWelf_{welfare_fn.__name__}'
        if tau is not None:
            self.name += f'_tau={tau}'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
            unprotected_vals, positive_pred, dict_of_nonclass_attrs):

        # Compute welfare for each sensitive class
        prot_welf, unprot_welf = self.calc_welfare(actual,
                                                   predicted,
                                                   dict_of_sensitive_lists,
                                                   single_sensitive_name,
                                                   unprotected_vals,
                                                   positive_pred,
                                                   dict_of_nonclass_attrs)
        if self.transform_welf is not None:
            prot_welf = self.transform_welf(prot_welf)
            unprot_welf = self.transform_welf(unprot_welf)

        if self.tau is not None:
            prot_welf = prot_welf >= self.tau
            unprot_welf = unprot_welf >= self.tau

        mean_prot_welf = np.mean(prot_welf)
        mean_unprot_welf = np.mean(unprot_welf)

        # print(f'mean_prot_welf: {mean_prot_welf}')
        # print(f'mean_unprot_welf: {mean_unprot_welf}')

        demparwelf =  np.abs(mean_prot_welf / mean_unprot_welf)

        return demparwelf

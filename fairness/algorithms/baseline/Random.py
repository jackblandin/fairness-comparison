import numpy as np

from fairness.algorithms.baseline.Generic import Generic


class Random(Generic):
    """
    Randomly predicts positive or negative class with probability equal to the
    positi. If it's predicting a negative, it samples from the non-positive class
    values with equal probability.
    """
    def __init__(self):
        Generic.__init__(self)
        self.name = "Random"

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):

        mean_pos_rate = np.mean(train_df[class_attr] == positive_class_val)
        pos_idx = np.random.rand(len(test_df)) < mean_pos_rate

        preds = np.empty(len(test_df), dtype=train_df[class_attr].dtype)
        for idx, is_pos in enumerate(pos_idx):
            try:
                if is_pos:
                    preds[idx] = positive_class_val
                else:
                    sample = positive_class_val
                    # Sample a non-positive class value
                    while sample == positive_class_val:
                        sample = train_df[class_attr].sample(1).values[0]
                    preds[idx] = sample
            except Exception as e:
                raise e

        return preds, []

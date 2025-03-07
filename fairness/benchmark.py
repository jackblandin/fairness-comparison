import fire
import os
import pandas as pd
import pathlib
import statistics
import sys


from fairness import results
from fairness.data.objects.list import DATASETS, get_dataset_names
from fairness.data.objects.ProcessedData import ProcessedData
from fairness.algorithms.list import ALGORITHMS
from fairness.metrics.list import get_metrics

from fairness.algorithms.ParamGridSearch import ParamGridSearch

NUM_TRIALS_DEFAULT = 10

def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for a in result:
        print("  %s" % a)
    return result

def run(num_trials = NUM_TRIALS_DEFAULT, dataset = get_dataset_names(),
        algorithm = get_algorithm_names()):
    algorithms_to_run = algorithm

    # L0: For each dataset
    print("Datasets: '%s'" % dataset)
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("\nEvaluating dataset:" + dataset_obj.get_dataset_name())

        processed_dataset = ProcessedData(dataset_obj)
        train_test_splits = processed_dataset.create_train_test_splits(num_trials)

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
        # L1: For each sensitive attribute
        for sensitive in all_sensitive_attributes:

            print("Sensitive attribute:" + sensitive)

            detailed_files = dict((k, create_detailed_file(
                                          dataset_obj.get_results_filename(sensitive, k),
                                          dataset_obj,
                                          processed_dataset.get_sensitive_values(k), k))
                for k in train_test_splits.keys())

            # L2: For each algorithm
            for algorithm in ALGORITHMS:
                if not algorithm.get_name() in algorithms_to_run:
                    continue

                print("    Algorithm: %s" % algorithm.get_name())
                print("       supported types: %s" % algorithm.get_supported_data_types())
                if algorithm.__class__ is ParamGridSearch:
                    # Optional: L3: For each parameter combination (I think this actually occurs in run_eval_alg)
                    param_files =  \
                        dict((k, create_detailed_file(
                                     dataset_obj.get_param_results_filename(sensitive, k,
                                                                            algorithm.get_name()),
                                     dataset_obj, processed_dataset.get_sensitive_values(k), k))
                          for k in train_test_splits.keys())
                # L3: For each trial
                for i in range(0, num_trials):
                    print(f'\nTrial {i+1}')
                    # L4: For each supported data type
                    for supported_tag in algorithm.get_supported_data_types():
                        train, test = train_test_splits[supported_tag][i]
                        try:
                            params, results, param_results =  \
                                run_eval_alg(algorithm, train, test, dataset_obj, processed_dataset,
                                             all_sensitive_attributes, sensitive, supported_tag, i, dataset_obj)
                        except Exception as e:
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                            print("Failed: %s" % e, file=sys.stderr)
                        else:
                            write_alg_results(detailed_files[supported_tag],
                                              algorithm.get_name(), params, i, results)
                            if algorithm.__class__ is ParamGridSearch:
                                for params, results in param_results:
                                    write_alg_results(param_files[supported_tag],
                                                      algorithm.get_name(), params, i, results)

            print("Results written to:")
            for supported_tag in algorithm.get_supported_data_types():
                print("    %s" % dataset_obj.get_results_filename(sensitive, supported_tag))

            for detailed_file in detailed_files.values():
                detailed_file.close()

def write_alg_results(file_handle, alg_name, params, run_id, results_list):
    line = alg_name + ','
    params = ";".join("%s=%s" % (k, v) for (k, v) in params.items())
    line += params + (',%s,' % run_id)
    line += ','.join(str(x) for x in results_list) + '\n'
    file_handle.write(line)

def write_train_test_datasets(alg_name, params, run_id, train_df, test_df, tag, dataset_obj, single_sensitive):
    """Writes X, y, yhat to CSVs."""
    filename = dataset_obj.get_dataset_name() + '_' + single_sensitive + '_' + tag + '_' + alg_name + '_'
    if len(params.items()) > 0:
        params = ";".join("%s=%s" % (k, v) for (k, v) in params.items())
        filename += params + '_'
    filename += str(run_id)
    home = pathlib.Path.home()
    path = str(home / '.fairness/train_test_datasets')
    train_fp = path+'/'+filename+'_train.csv'
    test_fp = path+'/'+filename+'_test.csv'
    train_df.to_csv(train_fp, index=None)
    test_df.to_csv(test_fp, index=None)

def run_eval_alg(algorithm, train, test, dataset, processed_data, all_sensitive_attributes,
                 single_sensitive, tag, run_id, dataset_obj):
    """
    Runs the algorithm and gets the resulting metric evaluations.
    """
    privileged_vals = dataset.get_privileged_class_names_with_joint(tag)
    positive_val = dataset.get_positive_class_val(tag)

    # get the actual classifications and sensitive attributes
    actual = test[dataset.get_class_attribute()].values.tolist()
    sensitive = test[single_sensitive].values.tolist()

    predicted, params, predictions_list =  \
        run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
                privileged_vals, positive_val, run_id, tag, dataset_obj)

    # make dictionary mapping sensitive names to sensitive attr test data lists
    dict_sensitive_lists = {}
    dict_nonclass_lists = {}
    for sens in all_sensitive_attributes:
        dict_sensitive_lists[sens] = test[sens].values.tolist()
        # get the nonsensitive attributes (used for welfare/cost calculations)
        dict_nonclass_lists[sens] = dataset.get_nonclass_attribute_values(sens, test)

    sensitive_dict = processed_data.get_sensitive_values(tag)

    one_run_results = []
    for metric in get_metrics(dataset, sensitive_dict, tag):
        result = metric.calc(actual, predicted, dict_sensitive_lists, single_sensitive,
                privileged_vals, positive_val, dict_nonclass_lists)
        one_run_results.append(result)

    # handling the set of predictions returned by ParamGridSearch
    results_lol = []
    if len(predictions_list) > 0:
        for param_name, param_val, predictions in predictions_list:
            params_dict = { param_name : param_val }
            results = []
            for metric in get_metrics(dataset, sensitive_dict, tag):
                result = metric.calc(actual, predictions, dict_sensitive_lists, single_sensitive,
                        privileged_vals, positive_val, dict_nonclass_lists)
                results.append(result)
            results_lol.append( (params_dict, results) )

    return params, one_run_results, results_lol

def run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
            privileged_vals, positive_val, run_id, tag, dataset_obj):
    class_attr = dataset.get_class_attribute()
    params = algorithm.get_default_params()

    # Note: the training and test set here still include the sensitive attributes because
    # some fairness aware algorithms may need those in the dataset.  They should be removed
    # before any model training is done.
    # NOTE JDB 12/10/2021: predictions_list seems to always be [] in every model
    predictions, predictions_list =  \
        algorithm.run(train, test, class_attr, positive_val, all_sensitive_attributes,
                      single_sensitive, privileged_vals, params)

    train_df = pd.DataFrame(train).copy()
    test_df = pd.DataFrame(test).copy()
    test_df['preds'] = predictions

    write_train_test_datasets(algorithm.get_name(), params, run_id, train_df, test_df, tag, dataset_obj, single_sensitive)

    return predictions, params, predictions_list


def get_dict_sensitive_vals(dict_sensitive_lists):
    """
    Takes a dictionary mapping sensitive attributes to lists in the test data and returns a
    dictionary mapping sensitive attributes to lists containing each sensitive value only once.
    """
    newdict = {}
    for sens in dict_sensitive_lists:
         sensitive = dict_sensitive_lists[sens]
         newdict[sens] = list(set(sensitive))
    return newdict

def create_detailed_file(filename, dataset, sensitive_dict, tag):
    return results.ResultsFile(filename, dataset, sensitive_dict, tag)
    # f = open(filename, 'w')
    # f.write(get_detailed_metrics_header(dataset, sensitive_dict, tag) + '\n')
    # return f

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree
import argparse
import pdb
import numpy as np
from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1
)
from joblib import dump, load
from itertools import product

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--clf_name', type=str, default='svm', help='model name')
    parser.add_argument('--random_state', type=int, default=123, help='random seed')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_option()
    
    # 1. set the ranges of hyper parameters
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


    #params= list(product(gamma_list,c_list))
    svm_params = {}
    svm_params["gamma"] = gamma_list
    svm_params["C"] = c_list
    svm_h_param_comb = get_all_h_param_comb(svm_params)

    max_depth_list = [2, 10, 20, 50, 100]

    dec_params = {}
    dec_params["max_depth"] = max_depth_list
    dec_h_param_comb = get_all_h_param_comb(dec_params)

    h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}

    # PART: load dataset -- data from csv, tsv, jsonl, pickle
    digits = datasets.load_digits()
    data_viz(digits)
    data, label = preprocess_digits(digits)
    # housekeeping
    del digits

    # define the evaluation metric
    metric_list = [metrics.accuracy_score, macro_f1]
    h_metric = metrics.accuracy_score


    n_cv = 1
    results = {}
    for n in range(n_cv):
        state = args.random_state
        x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
            data, label, train_frac, dev_frac,state
        )
        # PART: Define the model
        # Create a classifier: a support vector classifier
        
        if args.clf_name == "svm":
            models_of_choice = {
            "svm": svm.SVC()
            
            }
            
        else: 
            models_of_choice = {
            "decision_tree": tree.DecisionTreeClassifier()
            }
        
        for clf_name in models_of_choice:
            clf = models_of_choice[clf_name]
            print("[{}] Running hyper param tuning for {}".format(n,clf_name))
            actual_model_path = tune_and_save(
                clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=None
            )

            # 2. load the best_model
            best_model = load(actual_model_path)

            # PART: Get test set predictions
            # Predict the value of the digit on the test subset
            predicted = best_model.predict(x_test)
            if not clf_name in results:
                results[clf_name]=[]    

            results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
            # 4. report the test set accurancy with that best model.
            # PART: Compute evaluation metrics
            print(
                f"Classification report for classifier {clf}:\n"
                f"{metrics.classification_report(y_test, predicted)}\n"
            )
            print(results)
            final_restults = {
                'test_accuracy':results[args.clf_name][0]['accuracy_score'],
                'macro_f1':results[args.clf_name][0]['macro_f1'],
                'model path': actual_model_path
            }
            
            f_name = 'results/'+str(args.clf_name)+'_'+str(args.random_state)+'.txt'
            with  open(f_name,'w') as f:
                f.write(str(final_restults))

            


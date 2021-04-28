from lifelines import CoxPHFitter,GeneralizedGammaRegressionFitter,WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
import numpy as np
import pandas as pd
from lifelines.datasets import load_rossi

t_var = "week"
target = "arrest"
data_ = load_rossi()
penaliser=0.01
l1_ratio=0.99
for mdls in [WeibullAFTFitter(fit_intercept =True,penalizer=penaliser,l1_ratio=l1_ratio),LogNormalAFTFitter(fit_intercept=True,penalizer=penaliser,l1_ratio=l1_ratio), LogLogisticAFTFitter(penalizer =penaliser),CoxPHFitter(penalizer=penaliser,l1_ratio=l1_ratio)]:#,GeneralizedGammaRegressionFitter(penalizer=penaliser,l1_ratio=l1_ratio)]:
    print(mdls)

    data__ = data_#data_[data_["arrest"]==1]
    timelimes = sorted(data__[t_var].unique())
    mdl = estimator
    mdl.fit(data__,t_var, y_var)#,timeline =timelimes)
    ci = mdl.score(data__,scoring_method="concordance_index")
    print(ci)

    # Pick only censored data
    surv_data = data_[data_[y_var]==0]
    t_var_ = t_var
    target = y_var
    surv_data_last_obs = surv_data[t_var_]



    predict_expectation = surv_data.loc[:,[target,t_var_]]
    try:
        predict_expectation.loc[:,"expected"] = mdl.predict_expectation(surv_data)
    except:
        predict_expectation.loc[:,"expected"] = mdl.predict_expectation(surv_data,conditional_after =surv_data_last_obs )


    predict_expectation.loc[:,"diff_expectation"] = predict_expectation.loc[:,"expected"] - predict_expectation.loc[:,t_var_] 

    predict_median = mdl.predict_median(surv_data, conditional_after =surv_data_last_obs)
    predict_expectation.loc[:,"median"] = predict_median
    predict_expectation.loc[:,"diff_median"] = predict_expectation.loc[:,"median"] - predict_expectation[t_var_] 

    mean_diff_median_method = predict_expectation.loc[predict_expectation["diff_median"]!=np.inf,"diff_median"].mean()
    mean_diff_predict_expectation = predict_expectation["diff_expectation"].mean()
    display(predict_expectation)
    print(f"{np.round(mean_diff_predict_expectation,3)}")
    print(f"{np.round(mean_diff_median_method,3)}")
    print("\n")


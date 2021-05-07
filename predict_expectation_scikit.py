import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.datasets import load_flchain,load_breast_cancer,load_aids, load_whas500, load_veterans_lung_cancer
from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold, GridSearchCV, LeaveOneOut


def create_x_y(df,target_var="target",time_var="time"):
    x_cols = df.columns[~df.columns.isin([target_var,time_var])]
    # Get x,y in the right format
    X = df[x_cols]
    y = df[[target_var,time_var]]
    y[target_var] = y[target_var].astype(bool)
    y = y.to_records(index=False)
    return X,y


def prodict_compare_survival_times(fitted_estimate,x,y):
    '''
    Get the median of each surv function and add it to a dataframe for expected time comparison
    
    '''
    counter = 0
    surv_funcs = {"surv":fitted_estimate.predict_survival_function(x)}
    ids = pd.Series(x.index)
    predictions_surv = pd.DataFrame(index=ids)

    for key, value in surv_funcs.items():
        for fn in value:
            id = ids.iloc[counter]
            actual_time = y[time][counter]
            actual_target = y[target][counter]
            times_ = pd.Series(fn.x,name="times")
            survival = pd.Series(fn(fn.x),name="survival")
            a = pd.concat([times_,survival],axis=1)
            b = a[a["survival"]<=0.5].sort_values("survival",ascending=False)
            predictions_surv.loc[id,"actual_times"] = actual_time
            if b.empty:
                predictions_surv.loc[id,"predicted_time"] = np.inf
            else:
                prediction = b.iloc[0]["times"]
                predictions_surv.loc[id,"predicted_time"] = prediction           
            
            predictions_surv.loc[id,target] = actual_target        
            counter += 1

    predictions_surv["diff"] = predictions_surv["predicted_time"] - predictions_surv["actual_times"]
    mean_diff = predictions_surv["diff"].mean()
    actuals_median = predictions_surv["actual_times"].median()
    actuals_mean =predictions_surv["actual_times"].mean()
    average_predicted_median = predictions_surv.loc[predictions_surv["predicted_time"]!=np.inf,"predicted_time"].mean(skipna=True)
    results = pd.Series([mean_diff,actuals_median,actuals_mean,average_predicted_median],index=["Mean_diff","actual_median","actuals_mean","mean_of_predicted_median"])

    return predictions_surv,results


##############################################################
# Get data and format it for scikit survival and lifelines
# toy dataset
X, y = load_whas500()
X = X.astype(float)
target = y.dtype.names[0]
time = y.dtype.names[1]
y[time] = np.ceil(y[time]/30)
t_var = pd.Series(y[time],name=time)
y_var = pd.Series(y[target],name=target).astype(int)
df = pd.concat([X,y_var,t_var],axis=1)
time_points = np.array(sorted(np.unique(t_var)))

censored_x, censored_y = create_x_y(df_censored,target,time)
uncensored_x, uncensored_y = create_x_y(df_uncensored,target,time)


#################################################
# Quick fit and CV to check  how good the dataset is without any changes
rsf = RandomSurvivalForest(n_estimators=200)
gbs = GradientBoostingSurvivalAnalysis(n_estimators=200)
cns = CoxnetSurvivalAnalysis(l1_ratio=0.90, fit_baseline_model=True)

# Check the concordance index is good enough
cv = KFold(n_splits=2,shuffle=True,random_state=0)

rsf_score = cross_val_score(rsf,X,y,cv=cv)
gbs_score = cross_val_score(gbs,X,y,cv=cv)
cns_score = cross_val_score(cns,X,y,cv=cv)
print(rsf_score,gbs_score,cns_score)



######################################################
# Try predicting on censored/uncensored data
######################################################

## Fit on whole data
rsf.fit(X,y)
# Check how good it is at classifiying the risk scores in the observable data
rsf.score(uncensored_x,uncensored_y)

#predict on uncensored
oredicted_time_whole_uncensored, results_uncensored = prodict_compare_survival_times(rsf,uncensored_x,uncensored_y)
#predict on censored (what makes sense)
predicted_time_whole_censored, results_censored = prodict_compare_survival_times(rsf,censored_x,censored_y)
#predict on whole
predicted_time_whole, results_whole = prodict_compare_survival_times(rsf,X,y)


# Fit on uncensored data
rsf.fit(uncensored_x,uncensored_y)
## predict on uncensored
predicted_time_uncensored_uncensored, results_uncensored_ = prodict_compare_survival_times(rsf,uncensored_x,uncensored_y)






#########################################################
# Same but in lifelines
##############un###########################################
from lifelines.utils import find_best_parametric_model
from lifelines.plotting import qq_plot
from lifelines import GeneralizedGammaRegressionFitter,PiecewiseExponentialRegressionFitter,WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter, AalenAdditiveFitter, LogNormalFitter,SplineFitter, GeneralizedGammaFitter,WeibullFitter,LogLogisticFitter,ExponentialFitter,PiecewiseExponentialFitter

def prodict_compare_survival_times_lifelines(fitted_estimate,df,time="time",target="target"):
    '''
    Get the expected time and median of each surv function and add it to a dataframe for comparison
    '''
    surv_data_last_obs = df[time]
    predict_expectation = df.loc[:,[target,time]]

    try:
        predict_expectation.loc[:,"expected"] = fitted_estimate.predict_expectation(df,conditional_after =surv_data_last_obs )
    except:
        predict_expectation.loc[:,"expected"] = fitted_estimate.predict_expectation(df)
    

    predict_expectation.loc[:,"diff_expectation"] = predict_expectation.loc[:,"expected"] - predict_expectation.loc[:,time] 

    predict_median = fitted_estimate.predict_median(df, conditional_after =surv_data_last_obs)
    predict_expectation.loc[:,"median"] = predict_median
    predict_expectation.loc[:,"diff_median"] = predict_expectation.loc[:,"median"] - predict_expectation[time] 

    mean_diff_median_method = predict_expectation.loc[predict_expectation["diff_median"]!=np.inf,"diff_median"].mean()
    mean_diff_predict_expectation = predict_expectation["diff_expectation"].mean()
    actuals_median = predict_expectation[time].median()
    actuals_mean = predict_expectation[time].mean()
    average_predicted_medians = predict_expectation.loc[predict_expectation["median"]!=np.inf,"median"].mean()
    results = pd.Series([mean_diff_median_method,mean_diff_predict_expectation,actuals_median,actuals_mean,average_predicted_medians],index=["mean_diff_median_method","mean_diff_predict_expectation","actuals_median","actuals_mean","average_predicted_medians"])
    print(f"{np.round(mean_diff_predict_expectation,3)}")
    print(f"{np.round(mean_diff_median_method,3)}")
    print("\n")

    return predict_expectation,results



# t_var = pd.Series(y[time],name=time)
# y_var = pd.Series(y[target],name=target).astype(int)
# df = pd.concat([X,y_var,t_var],axis=1)
# df_censored = df[df[target]==0]
# df_uncensored = df[df[target]==1]
# time_points = np.array(sorted(np.unique(t_var)))

# Find the best parametric model (These are not regression models)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.reshape(4,)
best_model, best_aic_ = find_best_parametric_model(df[time], df[target], scoring_method="AIC")
print(best_model)
for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter(),ExponentialFitter()]):
    model.fit(df[time], df[target])
    qq_plot(model, ax=axes[i])

# This changes depending on what "best_model" is
best_model.print_summary()


lnf = LogNormalAFTFitter()
lnf.fit(df,time,target)
lnf.concordance_index_
lnf.score(df,scoring_method="concordance_index")
lnf.score(df_uncensored,scoring_method="concordance_index")


    
#predict on uncensored
oredicted_time_whole_uncensored, results_uncensored = prodict_compare_survival_times_lifelines(lnf,df_uncensored,time,target)
#predict on censored (what makes sense)
predicted_time_whole_censored, results_censored = prodict_compare_survival_times_lifelines(lnf,df_censored,time,target)
#predict on whole
predicted_time_whole, results_whole = prodict_compare_survival_times_lifelines(lnf,df,time,target)


# Fit on uncensored data
lnf.fit(df_uncensored,time,target)
## predict on uncensored
predicted_time_uncensored_uncensored, results_uncensored_ = prodict_compare_survival_times_lifelines(lnf,df_uncensored,time,target)























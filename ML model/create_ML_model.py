#!/usr/bin/env python
# coding: utf-8

# In[2]:



# import pandas and matplotlib 
import pandas as pd 
import matplotlib.pyplot as plt 
from statistics import mean 
from statistics import stdev
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV
import seaborn as sns
from pymongo import MongoClient
from scipy.stats import uniform


import math
from datetime import date
from datetime import datetime
# create 2D array of table given above 
data = [['E001', 'M', 34, 123, 'Normal', 350], 
        ['E002', 'F', 40, 114, 'Overweight', 450], 
        ['E003', 'F', 37, None, 'Obesity', 169], 
        ['E004', 'M', 30, None, 'Underweight', 189], 
        ['E005', 'F', 44, 117, 'Underweight', 183], 
        ['E006', 'M', 36, 121, 'Normal', None ], 
        ['E007', 'M', 1000, 133, 'Obesity', 166], 
        ['E008', 'F', None, 140, 'Normal', 120], 
        ['E009', 'M', 32, 133, 'Normal', 75], 
        ['E010', 'M', 36, 133, 'Underweight', 40] ] 
  
# dataframe created with 
# the above data array 
dataframe = pd.DataFrame(data, columns = ['EMPID', 'Gender',  
                                    'Age', 'Sales', 
                                    'BMI', 'Income'] ) 
client = MongoClient('10.16.3.55', 27017)
db=client["testDB"]
processed_users=db["processed_entrepreneurs"]
dataset=list(processed_users.find({}))
df=pd.DataFrame(dataset)
df=df.drop(["following_ids","followers_ids"],axis=1)
df.head()


# In[ ]:



#The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that
# there is a strong positive correlation; When the coefficient is close to –1, it means
# that there is a strong negative correlation
# Finally, coefficients close to zero mean that there is no linear correlation.
# The correlation coefficient only measures linear correlations (“if x
# goes up, then y generally goes up/down”). It may completely miss
# out on nonlinear relationships
def get_linear_correlations(df, interested_columns=[],sort_column_name=None ):
    corr_matrix = df.corr()
    if(len(interested_columns)==0):
        if(sort_column_name!=None): 
            corr_matrix.sort_values(by=sort_column_name,ascending=True)
        return corr_matrix
    else:
        ans=corr_matrix[interested_columns]
        return ans
    


def merge_highly_correlated_columns(df,columns_that_dont_find_corelation,
                            threshold=0.83):
    temp=df.copy()
    corr_matrix=get_linear_correlations(df,df.columns)
    to_remove=set()
    for column_name in corr_matrix.columns:
        if(column_name in columns_that_dont_find_corelation):
            continue
        column=corr_matrix[column_name].drop(column_name)
        for index in column.index:
            if(index in columns_that_dont_find_corelation ):
                continue
            anapoda_onoma_plus=index+"_plus_"+column_name
            anapoda_onoma_minus=index+"_minus_"+column_name
            if(anapoda_onoma_plus in temp.columns):
                continue
            if(anapoda_onoma_minus in temp.columns):
                continue
            if(column[index]>threshold):
                new_column_name=column_name+"_plus_"+index
                temp[new_column_name]=(temp[column_name]+temp[index])/2
                to_remove.add(index)
                to_remove.add(column_name)
            elif(column[index]<(-threshold)):
                new_column_name=column_name+"_minus_"+index
                temp[new_column_name]=(temp[column_name]-temp[index])/2
                to_remove.add(index)
                to_remove.add(column_name)
    to_remove=list(to_remove)
    temp=temp.drop(to_remove,axis=1)
    return temp
   

def check_for_correlations_visually(df,promising_column_names):
    from pandas.plotting import scatter_matrix
    scatter_matrix(df[promising_column_names], figsize=(df.shape[0], 3*len(promising_column_names)),grid=True)

def set_missing_values_of_column_to_median(df, column_name):
    median = df[column_name].median()
    df[column_name]=df[column_name].fillna(median)
    return df
    
def set_all_missing_values_to_median(df):
    for column in df.columns:
        if(df[column].dtype=="int64" or df[column].dtype=="float64"):
            set_missing_values_of_column_to_median(df,column)


# In[ ]:


def find_average_of_list(lista):
    return mean(lista)
def find_standard_deviation(lista):
    return stdev(lista)

def detect_outlier_with_z_value(data_1):
    outliers=[]
    not_outliers=[]
    threshold=2.9
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for i,y in enumerate(data_1):
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append((i,y))
        else:
            not_outliers.append((i,y))
    return (outliers,not_outliers)
def IQR_score(lista):
    Q1 = lista.quantile(0.25)
    Q3 = lista.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    print(lista < (Q1 - 1.5 * IQR)) or (lista > (Q3 + 1.5 * IQR))
    
def detect_outliers_with_IQR(lista):
    outliers=[]
    sorted(lista)
    q1, q3= np.percentile(lista,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr) 
    upper_bound = q3 +(1.5 * iqr)
    for item in lista:
        if(item<lower_bound or item>upper_bound):
            outliers.append(item)
    return outliers
def find_indexes_of_not_outliers(not_outliers):
    indexes=[]
    for t in not_outliers:
        index=t[0]
        indexes.append(index)
    return indexes
def deteck_and_remove_outliers(df, column_name):
    outliers,not_outliers=detect_outlier_with_z_value(df[column_name])
    indexes=find_indexes_of_not_outliers(not_outliers)
    new_df=df[df.index.isin(indexes)]
#     new_df=df.loc[indexes]
    return new_df

def find_only_int_and_float_and_otherwise(df):
    s1=set()
    s2=set()
    dts = dict(df.dtypes)
    for key in dts.keys():
        if((dts[key].name in ["int64","float64","int32","float32"])):
            s1.add(key)
        else:
            s2.add(key)
    lista1=list(s1)
    lista2=list(s2)
    numbers_df = df[lista1].copy()
    otherwise_df=df[lista2].copy()
    return numbers_df,otherwise_df     
        

def normalize_df(dataframe):
    numbers_df,otherwise_df=find_only_int_and_float_and_otherwise(dataframe)
    scaler = MinMaxScaler()
    data=scaler.fit_transform(numbers_df)
    normalized_numbers_df=pd.DataFrame(data,columns=numbers_df.columns)
    normalized_numbers_df["_id"]=dataframe["_id"].astype(int)
    otherwise_df["_id"]=dataframe["_id"].astype(int)
    ans=otherwise_df.merge(normalized_numbers_df, on="_id")
    ans["_id"]=dataframe["_id"].astype(int)
    ans["got_funding"]=dataframe["got_funding"].astype(int)
    ans["_id"]=ans.pop("_id")
    ans["got_funding"]=ans.pop("got_funding")
    return ans

def standarize_df(df):
    numbers_df,otherwise_df=find_only_int_and_float_and_otherwise(df)
    scaler =  StandardScaler()
    data=scaler.fit_transform(numbers_df)
    standarized_number_df=pd.DataFrame(data,columns=numbers_df.columns)
    ans= {**otherwise_df,**standarized_number_df}
    ans["got_funding"]=df["got_funding"]
    ans["_id"]=df["_id"]
    ans["_id"]=ans.pop("_id")
    ans["got_funding"]=ans.pop("got_funding")
    return pd.DataFrame(ans)


# In[ ]:


# Just like the cross_cal_score() function, cross_val_predixt() performs K-fold cross-validation,
# but instead of returning the evaluation scores, it returns the predictions
# made on each test fold. This means that you get a clean prediction for each
# instance in the training set (“clean” meaning that the prediction is made by a model
# that never saw the data during training).

# from sklearn.model_selection import cross_val_predict
# y_train_pred = cross_val_predict(modelo, X_train, y_train_binary, cv=3)
#cross_val_score(modelo, X_train, y_train_binary, cv=3, scoring="accuracy")


# In[ ]:


def get_confustion_matrix(target_class,prediction_class):
    data=confusion_matrix(target_class, prediction_class)
    dataf=pd.DataFrame(data,columns=["target_class","prediction_class"])
    return dataf
def plot_confustion_matrix(conmax):
    plt.matshow(conmax, cmap=plt.cm.gray)
    plt.show()
def get_procision_and_recall(target_class,prediction_class):
    precision=precision_score(target_class,prediction_class)
    recall=recall_score(target_class, prediction_class)
    return precision,recall

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()
def show_precision_recall_curve(y_train_binary,y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_train_binary, y_scores)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
def plot_pressition_vs_recall(y_train_binary,y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_train_binary, y_scores)
    create_classsic_plot(precisions, recalls, column_name_x="precisions",
                         column_name_y="recalls", scatter=False, alpha=None)
def plot_roc_curve(y_train_binary,y_scores, label=None):
    fpr, tpr, thresholds = roc_curve(y_train_binary, y_scores)
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
def get_roc_auc_score(y_train_binary,y_scores):
    roc_auc_score=roc_auc_score(y_train_5, y_scores)
    return roc_auc_score


# In[ ]:


# One way to compare classifiers is to measure the area under the curve (AUC).
# A perfectclassifier will have a ROC AUC equal to 1, whereas a purely random classifier will
# have a ROC AUC equal to 0.5. 


# In[ ]:


def get_optimal_hyperparamteres_of_model_usig_grid_search (model,data,
    labels, param_grid, cv=5,scoring='f1',get_estimator=False):
    grid_search = GridSearchCV(model, param_grid, cv=cv,scoring=scoring)
    grid_search.fit(data, labels)
    if(get_estimator==False):
        return grid_search.best_params_
    else:
        return grid_search.best_estimator_


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
def get_optimal_hyperparamteres_of_model_using_randomized_search_CV (model,
    data,labels, param_grid, cv=5,scoring='f1',
    get_estimator=False):
    grid_search = RandomizedSearchCV(model, param_grid, cv=cv,
                        scoring=scoring,random_state=42)
    grid_search.fit(data, labels)
    if(get_estimator==False):
        return grid_search.best_params_
    else:
        return grid_search.best_estimator_


# In[ ]:


#checked
def SelectKBest_attributes(df,k=10,score_function=chi2):
    if(k>=len(df.columns)):
        return df
    X = df.iloc[:,0:-1]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
    if("_id" in X.columns):
        X=X.drop("_id",axis=1)
    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=score_function, k=k)
    bestfeatures.fit_transform(X,y)
    column_indexes=bestfeatures.get_support(["index"])
    new_columns=find_columns(df,column_indexes)
    if("got_funding" not in new_columns):
        new_columns.append("got_funding")
    if("_id" not in new_columns):
        new_columns.append("_id")
    new_df=df[new_columns]
    return new_df


# In[ ]:


#not_used-Not finished
def find_k_most_important_attributes(df,k=10):
    if(k>=len(df.columns)):
        return df
    X = df.iloc[:,0:-1]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
    feature_names = df.columns
    clf = LassoCV().fit(X, y)
    importance = np.abs(clf.coef_)
    idx_third = importance.argsort()[-(k+1)]
    threshold = (importance[idx_third] + 0.00001)*1.25
    sfm = SelectFromModel(clf, threshold=threshold)
    sfm.fit(X, y)
    sfm.transform(X)
    column_indexes=sfm.get_support(["index"])
    new_columns=find_columns(df,column_indexes)
    if("got_funding" not in new_columns):
        new_columns.append("got_funding")
    new_df=df[new_columns]
    return new_df


# In[ ]:


def find_columns(df, column_indexes):
    all_columns=df.columns
    new_columns=[]
    for index in column_indexes:
        new_columns.append(all_columns[index])
    return new_columns


# In[ ]:


#checked
def remove_features_with_low_variance(df):
    variances=df.var(axis=0)
    n=math.ceil((len(variances)/3))
    nsmallest=variances.nsmallest(n=n)
    threshold=nsmallest.mean()
    sel = VarianceThreshold(threshold=threshold)
    sel.fit_transform(df)
    column_indexes=sel.get_support(["index"])
    new_columns=find_columns(df,column_indexes)
    if("got_funding" not in new_columns):
        new_columns.append("got_funding")
    new_df=df[new_columns]
    return new_df


# In[ ]:


#checked
def make_heatmap_with_correlations(df,limit=10):
    #get correlations of each features in dataset
    corrmat = get_linear_correlations(df, interested_columns=[],sort_column_name="got_funding") 
    num=0
    if("_id"in df.columns):
        num=num+1
    if("got_funding"in df.columns):
        num=num+1
    if(limit>=len(df.columns)-num):
        most_important_features=df.columns
    else:
        most_important_features=find_most_important_attribute_names(df,n=limit)
    plt.figure()
    #plot heat map
    g=sns.heatmap(df[most_important_features].corr(),annot=True,cmap="RdYlGn",
                  linewidths=0.5)


# In[ ]:


#checked
def l1_l2_based_feaure_selection(df, c=0.01,penalty="l1",sparse_estimator="lsvc"):
    X = df.iloc[:,0:-1]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
    if(sparse_estimator=="lsvc"):
        lsvc = LinearSVC(C=0.01, penalty=penalty, dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(X)
        
    else:
        lreg = LogisticRegression(C=0.01, penalty="l2", dual=False).fit(X, y)
        model = SelectFromModel(lreg, prefit=True)
        X_new = model.transform(X)
    column_indexes=model.get_support(["index"])
    new_columns=find_columns(df,column_indexes)
    if("got_funding" not in new_columns):
        new_columns.append("got_funding")
    new_df=df[new_columns]
    return new_df


# In[ ]:


def Tree_based_feature_selection(df,n_estimators=50):
    X = df.iloc[:,0:-1]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
    clf = ExtraTreesClassifier(n_estimators=n_estimators)
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    model.transform(X)
    column_indexes=model.get_support(["index"])
    new_columns=find_columns(df,column_indexes)
    if("got_funding" not in new_columns):
        new_columns.append("got_funding")
    new_df=df[new_columns]
    return new_df


# In[ ]:


#checked
def Recursive_feature_elimination_with_cross_validation(df,c=1.0):
    # Build a classification task using 3 informative features
    X = df.iloc[:,0:-1]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(C=c,kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy')
    rfecv.fit_transform(X, y)
    column_indexes=rfecv.get_support(["index"])
    new_columns=find_columns(df,column_indexes)
    if("got_funding" not in new_columns):
        new_columns.append("got_funding")
    new_df=df[new_columns]
    return new_df


# In[ ]:


#checked
def Recursive_feature_elimination(df,c=1.0):
    X = df.iloc[:,0:-1]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range

    # Create the RFE object and rank each pixel
    svc = SVC(C=c,kernel="linear")
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit_transform(X, y)
    column_indexes=rfe.get_support(["index"])
    new_columns=find_columns(df,column_indexes)
    if("got_funding" not in new_columns):
        new_columns.append("got_funding")
    new_df=df[new_columns]
    return new_df


# In[ ]:


def select_features(df,k=10,c=0.1,list_of_eliminations=(1,1,1,1,1,1,1)):
    numbers_df,otherwise_df=find_only_int_and_float_and_otherwise(df)
    temp=numbers_df.to_dict()
    temp["got_funding"]=temp.pop("got_funding")
    new_df=pd.DataFrame(temp)
    if(list_of_eliminations[0]==1):
        new_df=remove_features_with_low_variance(new_df)
    if(list_of_eliminations[1]==1):
        new_df=l1_l2_based_feaure_selection(new_df,c=c)
    if(list_of_eliminations[2]==1):
        new_df=l1_l2_based_feaure_selection(new_df,sparse_estimator="any",c=c)
    if(list_of_eliminations[3]==1):
        new_df=Tree_based_feature_selection(new_df)
    if(list_of_eliminations[4]==1):
        new_df=Recursive_feature_elimination(new_df,c*10)
    if(list_of_eliminations[5]==1):
        new_df=Recursive_feature_elimination_with_cross_validation(new_df,c*10)
    if(list_of_eliminations[6]==1):
        try:
            new_df=SelectKBest_attributes(new_df, k=k)
        except:
            print("could do select k")

    ans=otherwise_df.join(new_df)
    if("got_funding"in df.columns):
        ans["got_funding"]=df["got_funding"]
        ans["got_funding"]=ans.pop("got_funding")
    if("_id"in df.columns):
        ans["_id"]=df["_id"]
        ans["_id"]=ans.pop("_id")
    return pd.DataFrame(ans)
    


# In[ ]:


def turn_boolean_to_int(df):
    new_df=df.copy()
    for column_name in df.columns:
        if(df[column_name].dtype=="bool"):
            new_df[column_name]=new_df[column_name].astype(int)
    return new_df


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

def categorize_column(df, column_name):
    
    new_column_name=column_name+"_category"
    df[new_column_name]=df[column_name]
    df[new_column_name].where(df[new_column_name]>5, other=0,inplace=True)
    df[new_column_name].where(df[new_column_name]<4000, other=5,inplace=True)
    df[new_column_name].where(df[new_column_name]<3000, other=4,inplace=True)
    df[new_column_name].where(df[new_column_name]<2000, other=3,inplace=True)
    df[new_column_name].where(df[new_column_name]<1000, other=2,inplace=True)
    df[new_column_name].where(df[new_column_name]<500, other=1,inplace=True)
    df[new_column_name].where(df[new_column_name]<=5, other=0,inplace=True)
    df["got_funding"]=df.pop("got_funding")
    return new_column_name

def is_able_to_make_samplified_set(df,column_category_name):
    freq=df[column_category_name].value_counts()
    flag=False
    for i in range(freq.max()):
        if(i not in freq):
            flag=True
            break
        elif(freq[i]<2):
            flag=True
            break
    if (flag==True):
        return False
    else:
        return True
    
def find_test_and_training_set_with_random_sampling(df,
                                            test_size=0.2):
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=42)
    return train_set, test_set
    
def find_test_and_training_set_with_stratified_sampling(df,column_name,
                                            test_size=0.2,n_splits=5):
    column_category_name=categorize_column(df, column_name)
    if(is_able_to_make_samplified_set(df,column_category_name)==False):
        print("unable to make samplified set, doing random instead")
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=42)
        return train_set, test_set
    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    for train_index, test_index in split.split(df, df[column_category_name]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set,strat_test_set
        


# In[ ]:


#A histogram shows the number of instances(on the vertical axis)
#that have a given value range (on the horizontal axis).
def create_histogram(df,column_names, figsize=None,limit=10):
    num=0
    if("_id"in df.columns):
        num=num+1
    if("got_funding"in df.columns):
        num=num+1
    if(limit>=len(df.columns)-num):
        most_important_features=df.columns
    else:
        most_important_features=find_most_important_attribute_names(df,n=limit)
    temp=df[most_important_features]
    temp.hist(column=temp.columns,figsize=figsize)
    plt.show()
    

def create_column_chart_all_columns(df,limit=10):
    num=0
    if("_id"in df.columns):
        num=num+1
    if("got_funding"in df.columns):
        num=num+1
    if(limit>=len(df.columns)-num):
        most_important_features=df.columns
    else:
        most_important_features=find_most_important_attribute_names(df,n=limit)
    temp=df[most_important_features]
    temp.plot.bar( align='center', alpha=0.5)


def create_column_chart_2_columns(df, x_aksonas_column_name, y_aksonas_column_name):
    plt.bar(df[x_aksonas_column_name], df[y_aksonas_column_name])
    plt.xlabel(x_aksonas_column_name)
    plt.ylabel(y_aksonas_column_name)
    plt.show()
    
def create_box_plot_chart_for_each_column(df):
    df.plot.box()
    
def create_box_plot_chart_for_one_column(df,column_name):
    plt.boxplot(df[column_name])
    plt.show()

def create_pie_chart(df, column_name,shadow=True,radius =1,
                labeldistance=1.1,pctdistance =0.6 ):
    plt.pie(df[column_name],
    autopct ='% 1.1f %%', shadow = shadow,radius =radius,
           labeldistance =labeldistance,pctdistance =pctdistance  )
    plt.show()


def create_scatter_plot(df, x_aksonas_column_name, y_aksonas_column_name):
    # scatter plot between income and age
    plt.scatter(df[x_aksonas_column_name], df[y_aksonas_column_name])
    plt.show()

#option to
# makes it much easier to visualize the places alpha
# 0.1where there is a high density of data points
def create_classsic_plot(column_x, column_y, column_name_x="x-axis",
                         column_name_y="y-axis", scatter=False, alpha=None):
    plt.xlabel(column_name_x)
    plt.ylabel(column_name_y)
    if (scatter == True):
        plt.scatter(column_x, column_y,alpha=alpha)
    else:
        plt.plot(column_x, column_y,alpha=alpha)

        
def create_scatter_plot_for_df(df,column_x_name, column_y_name,  alpha=None):
    df.plot(kind="scatter", x=column_x_name, y=column_y_name, alpha=alpha)

# The radius of each circle represents the value of radius_column(option s),
# and the color represents the color_column (option c).
# We will use a predefined color map (option cmap) called jet, which ranges from blue
# (low values) to red (high prices)
def advance_plot(df,column_x_name, column_y_name, radius_column_name,
                         color_column, alpha=None):
    df.plot(kind="scatter", x=column_x_name, y=column_y_name, alpha=alpha,
                 s=df[radius_column_name], label=radius_column_name,
                 c=color_column, cmap=plt.get_cmap("jet"), colorbar=True,
                 )
    plt.legend()


# In[ ]:


def make_scatter_plot_with_grouped_attributes(df,target_column_name, x_column_name,y_column_name):
    got_funded=df[df[target_column_name]==1]
    didnt_got_funded=df[df[target_column_name]==0]
    plt.scatter(got_funded[x_column_name], got_funded[y_column_name],c="blue")
    plt.scatter(didnt_got_funded[x_column_name], didnt_got_funded[y_column_name],c="red")
    plt.show()


# In[ ]:


def get_f1_score(target_class,prediction_class):
    f1=f1_score(target_class,prediction_class)
    return f1
from sklearn.metrics import fbeta_score

# y_pred is an array of predictions
def bestThresshold(y_true,y_pred):
    best_thresh = None
    best_score = 0
    for thresh in np.arange(0.1, 0.501, 0.01):
        score = f1_score(y_true, np.array(y_pred)>thresh)
        if score > best_score:
            best_thresh = thresh
            best_score = score
    return best_score, best_thresh

def get_bf1_score(target_class,prediction_class,beta=0.5):
    ans=fbeta_score(target_class, prediction_class,beta=beta)
    return ans


# In[ ]:


def display_scores(scores,median_tora,mean_tora,std_tora,fkati_tora):
    print("Scores:", scores)
    print("Median", median_tora)
    print("Mean:", mean_tora)
    print("Standard deviation:", std_tora)
    print("fkati:", fkati_tora)


#default scoring method should probably change giaa binary classification model.
from sklearn.model_selection import StratifiedKFold
def show_comparison_of_models_using_cross_validation(models,data,labels,
    scoring_method="f1",cv=5,column_category_name="got_funding",weight_of_median=0.5):
    
    
    best_mean_index=0
    best_mean=None
    best_std_index=0
    best_std=None
    best_median_index=0
    best_median=None
    best_fkati=0
    best_fkati_index=None
    for i,model in enumerate(models):
        scores = cross_val_score(model, data, labels,
        scoring=scoring_method, cv=cv)
#         rmse_scores = np.sqrt(scores)
        rmse_scores = scores

        median_tora=np.median(rmse_scores)
        mean_tora=rmse_scores.mean()
        std_tora=rmse_scores.std()
        fkati_tora=  (((1-weight_of_median)*mean_tora)+ (weight_of_median*median_tora))/std_tora
#         display_scores(rmse_scores,median_tora,
#                      mean_tora,std_tora,fkati_tora )
        if(best_mean==None or best_mean<mean_tora ):
            best_mean=mean_tora
            best_mean_index=i
        if(best_std==None or best_std>std_tora ):
            best_std=std_tora
            best_std_index=i
        if(best_median==None or best_median<median_tora ):
            best_median=median_tora
            best_median_index=i
        if(best_fkati==None or best_fkati<fkati_tora):
            best_fkati=fkati_tora
            best_fkati_index=i
            
    print("best_median ",best_median,"was with model ",best_median_index,
          "best mean ",best_mean,"was with model ",best_mean_index,
          "best_std ",best_std,"was with model ",best_std_index,
         "best_fkati ",best_fkati,"was with model ", best_fkati_index)
    
    return best_fkati_index
    
def find_most_important_attribute_names(df,n=1):
    temp=df.copy()
    n=n+1
    temp=select_features(s,list_of_eliminations=(0,0,0,0,0,0,1),k=n)
    if(n<=2):
        return temp.columns[0]
    else:
        return temp.columns[:n-1]
        


# In[ ]:


from sklearn.linear_model import LogisticRegression

def create_logistic_regression_model(x_train,y_train):
    # all parameters not specified are set to their defaults
    logisticRegr = LogisticRegression()
    param_grid=dict(C=[0,2,0.5,0.8],
    penalty=["l1", "l2", "elasticnet"],random_state=[42],max_iter=[20,50,100,150,200,250])
    best_estimator=get_optimal_hyperparamteres_of_model_using_randomized_search_CV (model=logisticRegr,
    data=x_train,labels=y_train, param_grid=param_grid, cv=cv,scoring='f1',
    get_estimator=True)
    best_estimator.fit(x_train, y_train)
    return best_estimator


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
def create_k_neighbors_classifier_model(x_train,y_train,n_neighbors=5):
    neigh = KNeighborsClassifier()
    param_grid=dict(weights=["uniform", "distance"],p=[1,2],leaf_size=[10,20,30,40,50,60,80,100],
                   n_jobs=[1,2,3,4],metric=["minkowski","euclidean","manhattan","chebyshev"],
                   algorithm=["auto","ball_tree","kd_tree"])
    best_estimator=get_optimal_hyperparamteres_of_model_using_randomized_search_CV (model=neigh,
    data=x_train,labels=y_train, param_grid=param_grid, cv=cv,scoring='f1',
    get_estimator=True)
    best_estimator.fit(x_train, y_train)
    return best_estimator


# In[ ]:


from sklearn.naive_bayes import GaussianNB
def create_gaussian_naive_bayes_model(x_train,y_train,max_depth=None,
                                     max_features=10):
    gnb = GaussianNB()
    param_grid=dict()
    best_estimator=get_optimal_hyperparamteres_of_model_using_randomized_search_CV (model=gnb,
    data=x_train,labels=y_train, param_grid=param_grid, cv=cv,scoring='f1',
    get_estimator=True)
    best_estimator.fit(x_train, y_train)
    return best_estimator
                                      


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
def create_decision_tree_classifier_model(x_train,y_train):
    clf = DecisionTreeClassifier(random_state=0)
    param_grid=dict(criterion=["gini","entropy"],splitter=["best"],
                  max_depth=[300,650,1000,None],min_samples_split=[1,2,3,5,8],
                   max_features=["auto","sqrt","log2",None],
                   random_state=[42],class_weight=["balanced",None])
    best_estimator=get_optimal_hyperparamteres_of_model_using_randomized_search_CV (model=clf,
    data=x_train,labels=y_train, param_grid=param_grid, cv=cv,scoring='f1',
    get_estimator=True)
    best_estimator.fit(x_train, y_train)
    return best_estimator


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def create_support_vector_classification_model(x_train, y_train):
    svc=SVC()
    param_grid=dict(C=[0.5,0.8,1,1.5,2,2.5],kernel=["linear","poly","rbf",
        "sigmoid" ],degree=[2,3,4],gamma=["scale","auto"],
        class_weight=["balanced",None],decision_function_shape=["ovo","ovr"],
                   break_ties=[True,False],random_state=[42])
    best_estimator=get_optimal_hyperparamteres_of_model_using_randomized_search_CV (model=svc,
    data=x_train,labels=y_train, param_grid=param_grid, cv=cv,scoring='f1',
    get_estimator=True)
    clf = make_pipeline(StandardScaler(), best_estimator)
    clf.fit(x_train, y_train)
    return clf


# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def create_SGDClasifier_model(x_train,y_train ):
    sgdc=SGDClassifier()
    param_grid=dict(loss=["hinge","log", "modified_huber", "squared_hinge", "perceptron"],
            penalty=["l1","l2"],alpha=[0.00005,0.0001,0.00015,0.0002,0.0003,0.001],
                    fit_intercept=[True],max_iter=[500,1000,1500,2000],epsilon=[0.05,0.1,0.2],
                     n_jobs=[1,2,3,4],learning_rate=["optimal"],warm_start=[False,True],
                    class_weight=["balanced",None],power_t=[0.2,0.4,0.5,0,7,0.9])
    best_estimator=get_optimal_hyperparamteres_of_model_using_randomized_search_CV (model=sgdc,
    data=x_train,labels=y_train, param_grid=param_grid, cv=cv,scoring='f1',
    get_estimator=True)
    clf = make_pipeline(StandardScaler(),best_estimator)
    clf.fit(x_train, y_train)
    return clf
    


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
def create_random_forest_model(x_train,y_train,n_estimators=100,
                max_depth=None):
    clf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,
                                 random_state=42)
    param_grid=dict(n_estimators=[50,100,150,200,250],criterion=["gini","entropy"],
                  max_depth=[300,650,1000,None],min_samples_split=[1,2,3,5,8],
                   max_features=["auto","sqrt","log2",None],oob_score=[False,True],
                   random_state=[42],class_weight=["balanced",None],n_jobs=[1,2,3,4],
                   )
    best_estimator=get_optimal_hyperparamteres_of_model_using_randomized_search_CV (model=clf,
    data=x_train,labels=y_train, param_grid=param_grid, cv=cv,scoring='f1',
    get_estimator=True)
    best_estimator.fit(x_train, y_train)
    return best_estimator


# In[ ]:


from sklearn.metrics import classification_report
def find_classifiction_report(predictions,y_test):
    report=classification_report(y_test, predictions,output_dict=True)
    return report


# In[ ]:


def does_he_have_at_least_one_friend_within_limit(user,relationship_collection,
                                friend_collection,max_friends_count):
    try:
        q=relationship_collection.find({"_id.is_followed":user["_id"]}).limit(max_friends_count)
        if(q!=None):
            relationships=list(q)
        else:
            relationships=[]
        ids=[]
        for relationship in relationships:
            ids.append(relationship["_id"]["is_followed_by"])
        q=friend_collection.find_one({"_id": {"$in": ids}})
        if(q!=None):
            return True
    except:
        print("something")
    try:
        q= relationship_collection.find({"_id.is_followed_by":user["_id"]}).limit(max_friends_count)
        if(q!=None):
            relationships=list(q)
        else:
            relationships=[]
        ids=[]
        for relationship in relationships:
            ids.append(relationship["_id"]["is_followed"])
        q=friend_collection.find_one({"_id": {"$in": ids}})
        if(q!=None):
            True
        else:
            return False
    except:
        return False


# In[ ]:


def find_final_dataset():
    client = MongoClient('10.16.3.55', 27017)
    db=client["testDB"]
    processed_entrepreneurs_collection=db["processed_entrepreneurs"]
    processed_friends_collection=db["w_processed_users_friends"]
    relationship_collection=db["relationships"]
    entrepreneurs=list(processed_entrepreneurs_collection.find({}))
    print("done with the list")
    final_sample=[]
    for i,entrepreneur in enumerate(entrepreneurs):
        if(does_he_have_at_least_one_friend_within_limit(entrepreneur,
                                    relationship_collection,
                                    processed_friends_collection,max_friends_count=150)==True):
            final_sample.append(entrepreneur)
        if(i%50==0):
            print("i=",i)
            
    return final_sample
        
        
        


# In[ ]:


client = MongoClient('10.16.3.55', 27017)
db=client["testDB"]
processed_users=db["processed_entrepreneurs"]
dataset=find_final_dataset()
df=pd.DataFrame(dataset)
df=df.drop(["following_ids","followers_ids","screen_name","statuses_count"],axis=1)
temp=df.copy()


# In[ ]:


temp.shape


# In[ ]:


s=turn_boolean_to_int(temp)
set_all_missing_values_to_median(s)
s=normalize_df(s)
s=select_features(s,list_of_eliminations=(1,0,0,1,0,0,1),k=22)
s=merge_highly_correlated_columns(df=s,columns_that_dont_find_corelation=["_id","got_funding"],
                            threshold=0.83)
q=s.copy()
s=s.drop(["_id","got_funding"],axis=1)
s["_id"]=q["_id"]
s["got_funding"]=q["got_funding"]
most_important_attribute_column_name=find_most_important_attribute_names(s,n=1)


# In[ ]:


list(s.columns)


# In[ ]:


# s=select_features(s,list_of_eliminations=(0,0,0,1,0,0,0),k=18)


# In[ ]:


s.shape


# In[ ]:


s.head()


# In[ ]:


create_histogram(s,s.columns,figsize=(15,15))


# In[ ]:


create_column_chart_all_columns(s,limit=3)


# In[ ]:


make_heatmap_with_correlations(s,limit=5)


# In[ ]:


s=s.drop(["_id"],axis=1)
    #check tous outliers
s=deteck_and_remove_outliers(s,most_important_attribute_column_name)#interesting_column_name


# In[ ]:


list(s.columns)


# In[ ]:


make_scatter_plot_with_grouped_attributes(df=s,target_column_name="got_funding",
        x_column_name="avg_friends_count_followers",y_column_name="avg_polarity_of_following_plus_avg_sentiment_of_following")


# In[ ]:


linerar_cor=get_linear_correlations(s)


# In[ ]:


linerar_cor["got_funding"]


# In[ ]:


s=s.drop("got_funding_category",axis=1)


# In[ ]:


list(s.columns)


# In[ ]:


if("got_funding_category"in s.columns):
    s=s.drop("got_funding_category",axis=1)

train_set,test_set=find_test_and_training_set_with_stratified_sampling(s,"got_funding")


# In[ ]:


x_train=train_set.iloc[:,:-1]
y_train=train_set.iloc[:,-1:]
x_test=test_set.iloc[:,:-1]
y_test=test_set.iloc[:,-1:]


# In[ ]:


x_train=x_train.drop("got_funding_category",axis=1)
x_test=x_test.drop("got_funding_category",axis=1)


# In[ ]:


list(x_train.keys())


# In[ ]:


#an xriazete ke edo. 
y_train=y_train.values.ravel()
y_test=y_test.values.ravel()


# In[ ]:


if(is_able_to_make_samplified_set(df,most_important_attribute_column_name)==False):
    cv=5
else:
    cv= StratifiedKFold(n_splits=5,random_state=42,shuffle=True)



# In[ ]:


logistic_reggresion_model=create_logistic_regression_model(x_train,y_train)
logistic_reggresion_model


# In[ ]:


k_neighbors_classifier_model=create_k_neighbors_classifier_model(x_train,y_train)
k_neighbors_classifier_model


# In[ ]:


gaussian_naive_bayes_model=create_gaussian_naive_bayes_model(x_train,y_train)
gaussian_naive_bayes_model


# In[ ]:


decision_tree_classifier_model=create_decision_tree_classifier_model(x_train,y_train)
decision_tree_classifier_model


# In[ ]:


support_vector_classification_model=create_support_vector_classification_model(x_train,y_train)
support_vector_classification_model


# In[ ]:


SGDClasifier_model=create_SGDClasifier_model(x_train,y_train)
SGDClasifier_model


# In[ ]:


random_forest_model=create_random_forest_model(x_train,y_train)
random_forest_model


# In[ ]:


models=[
    logistic_reggresion_model,
    k_neighbors_classifier_model,
       gaussian_naive_bayes_model,
       decision_tree_classifier_model,
       support_vector_classification_model,
       SGDClasifier_model,
       random_forest_model
       ]


# In[ ]:


best_fkati_index=show_comparison_of_models_using_cross_validation(
models,x_train,y_train,scoring_method="f1",cv=cv,column_category_name="got_funding",
weight_of_median=0.65)


# In[ ]:


best_model=models[best_fkati_index]


# In[ ]:


final_model=best_model.fit(x_train,y_train)


# In[ ]:


predictions=final_model.predict(x_test)


# In[ ]:


get_f1_score(y_test,predictions)


# In[ ]:


for model in models:
    model.fit(x_train,y_train)


# In[ ]:


for index,model in enumerate(models):
    predictions=model.predict(x_test)
    report=find_classifiction_report(y_test,predictions)["weighted avg"]
    precision=report["precision"]
    recall=report["recall"]
    f1_scores=report["f1-score"]
    support=report["support"]
    print("model",index,"score","\n\nrecall:",recall,"\nprecision",precision
         ,"\nf1_score",f1_scores,"\nsupport",support) 


# In[ ]:


report=find_classifiction_report(y_test,predictions)


# In[ ]:


report.keys()


# In[ ]:


report["weighted avg"]


# In[ ]:





# In[ ]:





# In[ ]:


predictions=models[4].predict(x_test)


# In[ ]:


y_test


# In[ ]:


f1_scores=find_classifiction_report(y_test,predictions)


# In[ ]:


q=[]
for score in f1_scores


# In[ ]:


f1_scores["weighted avg"]["f1-score"]


# average=macro says the function to compute f1 for each label,
# and returns the average without considering the proportion
# for each label in the dataset.

# average=weighted says the function to compute f1 for each label, and returns the average considering the proportion for each label in the dataset.

# The support is the number of samples of the true response that lie in that class

# In[ ]:


client = MongoClient('10.16.3.55', 27017)
db=client["testDB"]
processed_followers_collection=db["w_followers"]
processed_following_collection=db["w_followings"]
processed_followers_ids=list(processed_followers_collection.find({},{"_id":1}))
processed_following_ids=list(processed_following_collection.find({},{"_id":1}))
print("done with this")


# In[ ]:


ids= set()
for processed_followers_id in processed_followers_ids:
    ids.add(processed_followers_id["_id"])
for processed_following_id in processed_following_ids:
    try:
        ids.add(processed_following_id["_id"])
    except:
        print(processed_following_id["_id"])
        continue


# In[ ]:


len(ids)


# In[ ]:


dic={}
for it in ids:
    dic[str(it)]=[]


# In[ ]:


len(dic.keys())


# In[ ]:


processed_tweets_friends=db["processed_tweets_friends_and_followers"]
all_processed_tweets_cur=processed_tweets_friends.find({},
            { "sentiment":1, "polarity":1, "subjectivity":1, 
             "retweet_count":1, "favorite_count":1, "hashtag_count":1,
             "user_mention_count":1, "is_quote":1,
             "is_reply":1,"user_id":1})


# In[ ]:


all_processed_tweets=list(all_processed_tweets_cur)


# In[ ]:


for i,tweet in enumerate(all_processed_tweets_cur):
    print("i=",i)
    user_id=str(tweet["user_id"])
    print(user_id)
    if(user_id in dic.keys()):
        dic[user_id].append(tweet)
    if(i==10):
        break
        


# In[ ]:



save_time=db["save_time"]
save_time.insert_one(dic)


# In[ ]:


dic["43905403"]


# In[ ]:





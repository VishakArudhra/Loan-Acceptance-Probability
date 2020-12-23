#!/usr/bin/env python
# coding: utf-8

# In[402]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes = True)


# In[403]:


liab2loan = pd.read_csv('Bank_Personal_Loan_Modelling.csv')


# In[404]:


liab2loan.shape


# In[405]:


liab2loan.head()


# ###### From the above dataset, nominally every feature seems valubale except for two namely: ID & ZIP Code 

# In[406]:


liab2loan.drop(columns = ['ID','ZIP Code'], axis = 1, inplace = True)


# In[407]:


liab2loan.head()


# In[408]:


liab2loan.info()


# ###### Every feature seems to be either mostly integer type, float otherwise. Only two features seems to be categorical variables which are Education and Family. Both the variables seem to be already ordinally encoded. So none of the two categorical variables need encoding.  

# In[409]:


target_feat = liab2loan['Personal Loan']


# In[410]:


liab2loan.drop('Personal Loan',axis = 1, inplace = True)


# In[411]:


target_feat.head(10)


# In[412]:


liab2loan = pd.concat([liab2loan,target_feat],axis=1)


# In[413]:


liab2loan.head()


# ###### Here I've just pushed the target variable to the last for easy comprehension. 

# In[414]:


liab2loan.describe()


# ###### The above observation has come useful in tracing out a major junk data in the 'experience' column which is its Min value of '-3'. Also the standard deviations of all columns except 'Age', 'Experience', 'Income' and 'Mortgage' seem to be acceptable. The excepted columns need more attention and overall, all the columns need inspection for dirty values such  as '-3' in experience. 

# In[415]:


liab2loan.isnull().values.any()


# ###### There doesn't seem to be any empty value in any of the columns. 

# ### TASK 2.

# In[416]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.hist(liab2loan['Experience'],color = 'blue', edgecolor = 'black', bins = int(180/5));
plt.title('Experience Distribution');
plt.xlabel('experience');
plt.subplot(1,2,2)
sns.stripplot(liab2loan['Personal Loan'], liab2loan['Experience'])


# ###### From the distribution, 'Experience' seems to have a slight right skew. People seem to avail loan(or not) from all levels of experience. 

# In[417]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.hist(liab2loan['Age'],color = 'blue', edgecolor = 'black', bins = int(180/5));
plt.title('Age Distribution');
plt.xlabel('Age');
plt.subplot(1,2,2)
sns.stripplot(liab2loan['Personal Loan'], liab2loan['Age'])


# ###### From the distribution, 'Age' seems to have a slight left skew. From the strip plot it nearly appears as if only the people older than a certain age and younger than a certain age avail the loan. 

# In[418]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.hist(liab2loan['Income'],color = 'blue', edgecolor = 'black', bins = int(180/5));
plt.title('Income Distribution');
plt.xlabel('Income');
plt.subplot(1,2,2)
sns.stripplot(liab2loan['Personal Loan'], liab2loan['Income'])


# ###### Income's distribution has a heavy left skew. The strip plot suggests loanees only from above a certain level of income seem to avail the loan and people from all levels of income seem to not avail the loan. 

# In[419]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.hist(liab2loan['Mortgage'],color = 'blue', edgecolor = 'black', bins = int(180/5));
plt.title('Mortgage Distribution');
plt.xlabel('Mortgage');
plt.subplot(1,2,2)
sns.stripplot(liab2loan['Personal Loan'], liab2loan['Mortgage'])


# ###### The mortgage has a heavy left skew. House mortgage is shown to be majorly zero. Strip plot suggests that regardless of any level of house mortgage, loans are availed and also clearly that very less loans are availed than not. 

# In[420]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.countplot(liab2loan['Education']);
plt.title('Education Distribution');
plt.xlabel('Education');
plt.subplot(1,2,2)
sns.barplot(liab2loan['Education'],liab2loan['Personal Loan'])


# ###### From the count plot, it is observed that under graduates are higher in number than graduates and professionals who seem to be equal in numbers.   From the bar plot it is evident that professionals and graduates avail loans more than undergraduates.

# In[421]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.countplot(liab2loan['Securities Account']);
plt.title('Securities Account Distribution');
plt.xlabel('Securities Account');
plt.subplot(1,2,2)
sns.barplot(liab2loan['Securities Account'],liab2loan['Personal Loan'])


# ###### By the count plot, it is seen that very less people have securities account than not. From the bar plot it is seen than loan is awarded more to people with securities account than people with none. 

# In[422]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.countplot(liab2loan['CD Account']);
plt.title('CD Account Distribution');
plt.xlabel('CD Account');
plt.subplot(1,2,2)
sns.barplot(liab2loan['CD Account'],liab2loan['Personal Loan'])


# ###### CD account follows the same trend as of 'Securities Account'. 

# In[423]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.countplot(liab2loan['CreditCard']);
plt.title('Credit Card Distribution');
plt.xlabel('Credit Card');
plt.subplot(1,2,2)
sns.barplot(liab2loan['CreditCard'],liab2loan['Personal Loan'])

plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.countplot(liab2loan['Online']);
plt.title('Online Distribution');
plt.xlabel('Online');
plt.subplot(1,2,2)
sns.barplot(liab2loan['Online'],liab2loan['Personal Loan'])


# ###### Credit card barely follows the same trends as 'CD Account' and 'Securities account'. Although, the barplot nearly suggests that people with a card and people with none avail loans equally. 
# 
# ###### From the countplot, it is seen that more people are online than not.

# ### TASK 2 END 

# ### TASK 3.

# In[424]:


plt.figure(figsize=(6,5))
sns.countplot(liab2loan['Personal Loan'])
plt.title('Personal Loan Distribution')


# ###### Loanees are dramatically less in comparison with the non-loanees. Evidently enough the loanees by count in every aspect were lesser than non loanees. Such skewness or imbalance in the target variable can be really consequential during prediction and therefore "resampling" techniques can be taken up namely: "Oversampling", "Undersampling" or both. 

# # ||-------------------------DATA PREPROCESSING - Start-------------------------------------||

# In[425]:


liab2loan.boxplot(column = ['Experience','Age','Income','Mortgage'], return_type = 'axes', figsize = (8,8)) 


# ###### From the above observations, the following can be inferred: There seems to be quite a density of outliers in the income and mortgage  categories which I believe calls for futher statistical analysis. Moving ahead, the experience columns remains to be pruned of any negative values and overall, the above four categories require normalization.

# In[426]:


print(liab2loan.var())


# ###### The variance of the columns 'Age','Experience','Income' and 'Mortgage' seem exceedingly high and require cleaning. 

# In[427]:


liab2loan.skew()


# ### REMOVING NEGATIVE VALUES FROM 'EXPERIENCE'

# Outliers can be easily removed or replaced with a single line, but what to replace negatives with has to be decided. 

# In[428]:


liab2loan['Experience'].describe()


# In[429]:


plt.figure(figsize=(8,6))
sns.scatterplot(liab2loan['Age'],liab2loan['Experience'],alpha=0.7)
plt.show()


# ###### The relationship between Age and Experience seems very linear but a lot of 'Experience' below 30 years of age seems to be negative which calls for a fix. That given, we have two options to either go by mean or mode. First let's plot distribution for experience.

# In[430]:


sns.distplot(liab2loan['Experience'])


# ###### From the above plot, it appears that there's no reliable mode to replace negatives with, therefore mean would be a better option since it's in summary of all the observations including the modes. 

# In[431]:


liab2loan['Experience'][liab2loan['Age']<30].describe(include='all')


# In[432]:


target_labels = liab2loan['Experience'][liab2loan['Age']<30].mode().value_counts()
target_mode = liab2loan['Experience'][liab2loan['Age']<30].mode()



df = pd.concat([target_mode,target_labels],axis=0)


# In[433]:


print(df)


# ###### So here the counts of both the modes under the age of 30 seems unreliably low and therefore the mean would be a better replacement for negatives. 

# In[434]:


liab2loan['Experience'][liab2loan['Experience']<0] = 1.969262


# In[435]:


liab2loan.describe()


# ### OUTLIER IDENTIFICATION

# ###### Let's run an outlier test on the dataset and trace the columns to target: 

# In[436]:


feat_desc  = liab2loan.describe()
feat_desc


# In[437]:


outlier_list = []
for i in list(feat_desc):
    q1 = feat_desc[i]['25%']
    print(f'the q1 of {i} is {q1}')
    q3 = feat_desc[i]['75%']
    print(f'the q3 of {i} is {q3}')
    iqr =  abs(q1 - q3)
    print('.')
    print('.')
    print(f'the interquartile range of {i} is {iqr}')
    mx = feat_desc[i]['max']
    mn = feat_desc[i]['min']
    print('are there any outliers?')
    up_lim = q3 + 1.5*iqr
    dwn_lim = q1 - 1.5*iqr
    print('.')
    print('.')
    if ((mx<=up_lim) & (mn>=dwn_lim)):
        print(f'there are no outliers in {i}')
        
    else:
        print('whoop them outliers!')
        outlier_list.append(i)
    print('.')
    print('.')
    print('.')
    
    


# ###### Above is a routine which identifies columns from a dataset with outliers and collects them in a separate list. 

# In[438]:


feat_desc[outlier_list]


# ###### In the above generated data residue from the 'Outlier filteration', the only actionably valid columns Income, Mortgage and CCAvg, since all other columns seem to be normalized with z-score. 
# 

# In[439]:


outlier_list = outlier_list[0:3]


# In[440]:


outlier_list


# In[441]:


plt.figure(figsize = (6,5))
plt.hist(liab2loan['CCAvg'],color='blue',edgecolor='black',alpha = 0.7)
plt.xlabel('CCAvg Dist')


# ###### CCAvg evidently has a daunting left skew. 

# ## OUTLIER REMOVAL

# In[442]:


def remove_outliers(i):
    outliers=[]   #a fresh outlier list to accomodate outliers for each category column which later add to a df
    print('')
    print(f'Calculated outliers for {i}:')
    print('')
    q1 = feat_desc[i]['25%']
    q3 = feat_desc[i]['75%']
    iqr =  abs(q1 - q3)
    mx = feat_desc[i]['max']
    mn = feat_desc[i]['min']
    up_lim = q3 + 1.5*iqr
    dwn_lim = q1 - 1.5*iqr
    filterate = liab2loan[i][~((liab2loan[i]<dwn_lim)|(liab2loan[i]>up_lim))] #wipes outlier values to NaN values which will 
                                                                              #be treated later. 
    
        
    return filterate


# ###### Above is a function, which collects outliers for any column, filters them out and returns the column

# In[443]:


filterate = remove_outliers('Income')


# In[444]:


plt.figure(figsize=(15,3.5))
plt.subplot(1,2,1)
plt.hist(liab2loan['Income'],color = 'blue', edgecolor = 'black', alpha = 0.7)
plt.xlabel('Income w/ outliers')
plt.subplot(1,2,2)
plt.hist(filterate, color = 'red', edgecolor = 'black',alpha = 0.7)
plt.xlabel('Income w/o outliers')


# ###### As observed above, the lower and upper outliers seem to be removed generating null space in the column which will be filled later. 

# In[445]:


liab2loan['Income'] = filterate #income columns is replaced with refined data


# In[446]:


outlier_list


# In[447]:


filterate = remove_outliers('CCAvg')


# In[448]:


plt.figure(figsize = (15,3.5))
plt.subplot(1,2,1)
plt.hist(liab2loan['CCAvg'], color = 'blue',edgecolor = 'black', alpha = 0.7)
plt.xlabel('CCAvg w/ outliers')
plt.subplot(1,2,2)
plt.hist(filterate,color = 'red', edgecolor = 'black', alpha = 0.7)
plt.xlabel('CCAvg w/o outliers')


# In[449]:


liab2loan['CCAvg'] = filterate


# In[450]:


outlier_list


# In[451]:


filterate = remove_outliers('Mortgage')


# In[452]:


plt.figure(figsize=(15,3.5))
plt.subplot(1,2,1)
plt.hist(liab2loan['Mortgage'],color = 'blue',edgecolor='black',alpha=0.7)
plt.xlabel('CCAvg w/ outliers')
plt.subplot(1,2,2)
plt.hist(filterate,color='red',edgecolor='black',alpha=0.7)
plt.xlabel('CCAVG w/o outliers')


# In[453]:


liab2loan['Mortgage'] = filterate


# In[454]:


liab2loan.describe()


# In[455]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.hist(liab2loan['CD Account'], color = 'red',edgecolor = 'black',alpha = 0.7)
plt.xlabel('CD Account distribution')
plt.subplot(1,2,2)
plt.hist(liab2loan['Securities Account'], color = 'purple',edgecolor = 'black',alpha = 0.7)
plt.xlabel('Securities Account distribution')


# ###### Applying outlier removal even once could remove a whole category in these two columns. Therefore, probably some type of data skewness fixing technique could be employed here. 

# ### REMOVING MISSING VAUES

# In[456]:


liab2loan.isna().any()


# ###### A few features seem to have missing values since outlier removal. 

# In[457]:


inc_missing = pd.DataFrame(liab2loan.Income.isnull())
liab2loan[inc_missing['Income'] == True]


# In[458]:


mort_missing = pd.DataFrame(liab2loan.Mortgage.isna())
liab2loan[mort_missing['Mortgage'] == True]


# In[459]:


ccavg_missing = pd.DataFrame(liab2loan.CCAvg.isna())
liab2loan[ccavg_missing['CCAvg'] == True]


# ###### In the three tables above, the specific rows containing null/missing values respectively for each of three features can be observed. 

# In[460]:


print(inc_missing.shape)
print(mort_missing.shape)
print(ccavg_missing.shape)


# In[461]:


median_filler = lambda x:x.fillna(x.median())
liab2loan = liab2loan.apply(median_filler, axis =0)


# ###### Above is a routine run to replace NaN values with median in the respective columns. 

# In[462]:


liab2loan.isna().any()


# ###### Therefore, all the missing values have been filled with median. 

# ## Handling categorical data

# In[463]:


sns.barplot(liab2loan['Family'],liab2loan['Personal Loan'])


# In[464]:


sns.barplot(liab2loan['Education'],liab2loan['Personal Loan'])


# In[465]:


liab2loan['Family'] = liab2loan['Family'].astype(str)


# In[466]:


liab2loan['Family'].dtype


# In[467]:


liab2loan['Education'] = liab2loan['Education'].astype(str)


# In[468]:


liab2loan['Education'].dtype


# In[469]:


liab2loan.dtypes


# In[470]:


ordinal_encoder= OrdinalEncoder()


# In[471]:


ordinal_cat = ordinal_encoder.fit_transform(liab2loan[['Education','Family']])


# In[472]:


liab2loan[['Education','Family']] = ordinal_cat


# In[473]:


liab2loan.dtypes


# In[474]:


plt.figure(figsize=(8,5))
ht_mp = liab2loan.corr()
sns.heatmap(ht_mp)


# 

# ###### In the above heatmap, Personal Loan feature seems to correlate the most with Income. 

# # ||-------------------------DATA PREPROCESSING - END-------------------------------------||

# ### TASK 4.

# In[475]:


liab2loan.head()


# In[476]:


X = liab2loan.drop('Personal Loan',axis=1)
y = liab2loan['Personal Loan']


# In[477]:


X


# In[478]:


y


# In[479]:


X_scaled = X.apply(zscore)


# In[480]:


x_train, x_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.30,random_state=40)


# In[481]:


x_train


# In[482]:


y_train


# In[483]:


loanees = liab2loan.loc[liab2loan['Personal Loan']==1]
nolonees = liab2loan.loc[liab2loan['Personal Loan']==0]


# In[484]:


print(f'loanees number : {len(loanees)}, percentage: {round(len(loanees)/len(y),3)}')
print(f'nonlonees number : {len(nolonees)}, percentage: {round(len(nolonees)/len(y),3)}')


# In[485]:


print(f'train loanees number :{len(y_train[y_train == 1])}, percentage : {round(len(y_train[y_train == 1])/len(y),4)}')
print(f'train non loanees number :{len(y_train[y_train == 0])}, percentage : {round(len(y_train[y_train == 0])/len(y),4)}')


# In[486]:


print(f'test loanees number :{len(y_test[y_test == 1])}, percentage : {round(len(y_test[y_test == 1])/len(y),4)}')
print(f'test non loanees number :{len(y_test[y_test == 0])}, percentage : {round(len(y_test[y_test == 0])/len(y),4)}')


# ### TASK 5 and TASK 6 ( confusion matrix printing done simultaneously)

# ### PREDICTION USING : LOGISTIC, KNN & NAIVE BAYES

# In[487]:


x_train.describe()


# ### LOGISTIC REGRESSION 

# In[488]:


model = LogisticRegression(solver="liblinear")


# In[489]:


model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[490]:


mod_score = model.score(x_test,y_test)
print(mod_score)


# ###### TASK 6

# In[491]:


cm_log = metrics.confusion_matrix(y_test,y_pred,labels=[1,0])

df_cmlog = pd.DataFrame(cm_log, index = [i for i in['1','0']],columns = [i for i in ['pred_1','pred_0']])
plt.figure(figsize = (7,5))
sns.heatmap(df_cmlog, annot = True)


# In[492]:


log_report = metrics.classification_report(y_test,y_pred,labels=[1,0])
print(metrics.classification_report(y_test,y_pred,labels=[1,0]))


# ### NAIVE BAYES 

# In[493]:


nb_model = GaussianNB()
nb_model.fit(x_train,y_train.ravel())


# #### Performance in training set: 

# In[494]:


train_pred = nb_model.predict(x_train)

print(f'train accuracy: {round(metrics.accuracy_score(y_train,train_pred),4)}')


# #### Performance in the test set: 
# 

# In[495]:


y_pred = nb_model.predict(x_test)
print(f'test accuracy: {round(metrics.accuracy_score(y_test,y_pred),4)}')


# ###### TASK 6

# In[496]:


cm_nb = metrics.confusion_matrix(y_test,y_pred,labels=[1,0])
df_cmnb = pd.DataFrame(cm_nb, index = [i for i in ['1','0']], columns = [i for i in ['pred-1','pred-0']])
plt.figure(figsize=(7,5))
sns.heatmap(df_cmnb, annot = True)


# In[497]:


nb_report = metrics.classification_report(y_test,y_pred,labels=[1,0])
print(metrics.classification_report(y_test,y_pred,labels=[1,0]))


# ### KNN 

# In[498]:


knn_mod = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
knn_mod.fit(x_train,y_train)


# In[499]:


y_pred = knn_mod.predict(x_test)

knn_mod.score(x_test,y_test)


# ###### TASK 6 

# In[500]:


cm_knn_mod = metrics.confusion_matrix(y_pred,y_test,labels=[1,0])
df_cmknn = pd.DataFrame(cm_knn_mod, index = [i for i in ['1','0']], columns = [ i for i in ['pred- 1','pred - 0']])
plt.figure(figsize=(7,5))
sns.heatmap(df_cmknn, annot=True)


# In[501]:


knn_report = metrics.classification_report(y_test,y_pred,labels=[1,0])
print(metrics.classification_report(y_test,y_pred,labels=[1,0]))


# ###### An important inference to make from the confusion matrix heatmaps of the three models is that, the KNN model has commited the least "false negative" misclassifications proving higher accuracy among the models.  

# ### TASK 7

# In[502]:


print('logistic reg report')
print('')
print(log_report)
print('')
print('naive bayes report')
print('')
print(nb_report)
print('')
print('knn report')
print('')
print(knn_report)


# ###### Precision is the measure of the number of correct positive identifications or [True Positives] from all the samples idenitfied as POSITIVE.
# 
# ###### Recall is the measure of the correct positives identifications from the all the samples which are actually POSITIVE. 
# 
# ###### Therefore, of Precision and Recall, the latter would be a better measure of the model's correctness as: MORE  of 'Recall'/'Sensitivity' means less of false negatives.
# 
# ###### Less of False Negatives is very essential for the model's correctness as it would be a substantial loss to ignore a candidate who is very much a Potential Loanee.
# 
# ###### In this light, K NEAREST NEIGHBORS should be the most accurate or efficient model since, it's recall for '1' / 'One' is higher than the other two models. 

#     

#           

# ### WHY KNN WORKS BETTER THAN LOGISTIC REGRESSION AND NAIVE BAYES: 

# ###### LOGISTIC REGRESSION VS KNN:

# ###### KNN: 
# 
# Recognizes the "majority class" of all the samples at a selected distance from the target sample and on that basis determines the class. 
# 
# ###### LOGISTIC REGRESSION: 
# 
# Takes into account the various parameter values of the target sample and calculates a value to categorize based on the threshold. Only the latter is the more iterated step.
# 
# ##### Bulk of the calculation in KNN is only done post training which is 'distance based' classification as opposed to logistic regression where the same is rather during training. This creates a tradeoff between the two models: 
# 
# ###### IMPLEMENTATION: 
# 
# In the context of classification, the calculation is relatively very much simpler (although post training). Where (1) distances of all the surrounding points are calculated (2) the classes of only the points at a select distance are considered and (3) based on the majority class the target sample's class is determined. 
# 
# Therefore, the algorithm as taught is 
# 
# 1) immune to outliers. 
# 2) fairly easily implemented. 
# 3) Has no underlying assumption. 
# 
# ###### And so for the aforesaid reasons and the by the implied logic, KNN fares better at classification. 
# 
# ###### EFFICIENCY: 
# 
# Although, without time constraints KNN is a better deal, with time constraints (such as in 'real-time'), it's not the case. As already mentioned, the bulk of the calculations already happens during training (ignoring the occassional model update and maintenance) and therefore during the model execution, it's just a simple function of substitution. Therefore, in real-time, Logistic Regression is a better deal. 

# ### NAIVES BAYES VS THE OTHER TWO: 
# 
# ##### Why Naive Bayes does not catch up to LR  and KNN: 
# 
# The reason is simply that it's Naive; it assumes there'd be zero collinearity or zero correlation betw the features. If we plotted a pair-plot or/and a correlation chart among the features, we identify a correlation on some level betw any two features. 
# 
# ###### That is exactly why NB would perform much less than the LR and KNN. 

# In[503]:


plt.figure(figsize=(9,9))
sns.pairplot(liab2loan, diag_kind = 'kde')


# In[504]:


plt.figure(figsize=(12,10))
sns.heatmap(liab2loan.corr())


# As we can observe in the pairplot, there's a clear linear kind of correlation betw Age and Experience. 
# 
# and from the correlation heatmap, it's clear that Income and Credit card average are correlated and color-wise many tiles suggest correlation betw many feature pairs on various levels. 
# 
# ###### Therefore when it comes to classification, Naive Bayes shouldn't be in the first options. 

# Therefore, the best model for classification for the given dataset can be determined differently in two aspects: 
# 
# ###### by accuracy: 
# 
# KNN 
# 
# ###### by efficiency: 
# 
# LOGISTIC REGRESSION 
# 
# 

# In[ ]:





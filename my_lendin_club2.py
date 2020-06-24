


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Get dataset
df_acc = pd.read_csv('accepted.csv')
df_acc.head()

df_acc.columns

#Find columns that have maximum missing values
missing_fractions = df_acc.isnull().mean().sort_values(ascending=False)
df2 = (missing_fractions > 0.40) == True
df2 = pd.DataFrame(df2, columns = ['Binary1']).reset_index()
df_acc.drop(df2[df2['Binary1'] == True]['index'],axis = 1,inplace = True)

del missing_fractions

#Exploratory analysis
df_acc.info()
df_acc.describe()
df_acc.groupby('loan_status').count()
#Keep only Charged off and fully paid
df_acc = df_acc[(df_acc['loan_status'] == 'Charged Off') | (df_acc['loan_status'] == 'Fully Paid')]
df_acc['loan_status'] = df_acc['loan_status'].map({'Charged Off':0,'Fully Paid':1})
sns.countplot(x='loan_status',data = df_acc)
df_acc.groupby('loan_status').describe()

#Columns to keep on basis of lending data study of best known columns for potential investors

keep_list = ['addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'grade', 'home_ownership', 'id', 'initial_list_status', 'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title', 'total_acc', 'verification_status', 'zip_code']
drop_list = [r for r in df_acc.columns if r not in keep_list]
df_acc = df_acc.drop(drop_list, axis = 1)
del drop_list
del keep_list
del df2

#Id column
df_acc.groupby('id').count()
df_acc.drop(['id'],axis = 1,inplace = True)

#Now handle the rst of the missing values 
df_acc.isnull().mean().sort_values(ascending=False)
#emp_title,emp_length,mort_acc,title,revol_util,pub_rec_bankruptcies,dti,zip_code

#emp_title
df_acc['emp_title'].nunique()##378353 -- Too many unique values to deal
df_acc.drop(['emp_title'],axis = 1,inplace = True)
#emp_length
df_acc['emp_length'].nunique()
df_acc['emp_length'].isna().sum()#78511 rows
df_acc['annual_inc'].isna().sum()#0 rows
df_acc['annual_inc'].idxmax()
df_acc.iloc[539807]
#Loan Amount and status donot give a conclusive result for the employee length
df_acc[['loan_amnt','loan_status','emp_length']].groupby('emp_length').describe()
df_acc[['loan_status','emp_length']].groupby('emp_length').describe()

#Annual income and emp length sem to have a relationship
df_emp_len = df_acc[['annual_inc','emp_length']].groupby('emp_length').mean().reset_index()
df_emp_len['emp_length'] = df_emp_len['emp_length'].str.replace('year','').str.replace('s', '').str.replace(' ','').str.replace('<1','0').str.replace('+','')
df_emp_len['emp_length'] = df_emp_len['emp_length'].astype(int)
df_emp_len = df_emp_len.sort_values(by = ['emp_length'])
df_emp_len.columns
df_emp_len.to_numpy()
df_emp_len.annual_inc.values
df_acc['emp_length'].value_counts()
 
def fil_emp_lngth (var_emp_lngth,var_annual_inc):
    el = 0
    if len(str(var_emp_lngth)) == 3 :
        for r,a in enumerate(df_emp_len.annual_inc.values):
          if var_annual_inc <  a:
              el = r
              return el
          else:
              pass
        el = 0
    else:
        d = str(var_emp_lngth).replace('year','').replace('s', '').replace(' ','').replace('<1','0').replace('+','')
        return int(d)


df_acc['emp_new_lngth'] = df_acc.apply(lambda x:fil_emp_lngth(x['emp_length'],x['annual_inc']),axis = 1)

df_acc['emp_new_lngth'].unique()
df_acc['emp_new_lngth'].value_counts()
df_acc['emp_new_lngth'] = df_acc['emp_new_lngth'].fillna(10.0)#Only the ones above 84122
df_acc['emp_new_lngth'].isna().sum()#78511 rows
df_acc[df_acc['emp_new_lngth'].isna()]['annual_inc']
plt.figure(figsize =(12,5))
sns.lmplot(y = 'annual_inc', x = 'emp_new_lngth',data = df_acc)
sns.countplot( x = 'emp_new_lngth',data = df_acc)
df_acc[['loan_status','emp_new_lngth']].groupby('emp_new_lngth').describe()
df_acc.drop(['emp_length'],axis = 1,inplace = True)

#Mort_acc
df_acc['mort_acc'].unique()
df_acc['mort_acc'].value_counts()
df_acc.corr()['mort_acc'].sort_values(ascending = False) #Check if it is correlated to any other column
df_acc['total_acc'].unique()
df_acc['total_acc'].value_counts().sort_values(ascending = False)
#Lots of outliers... Its relation to others 
df_acc.corr()['total_acc'].sort_values(ascending = False) # It has positive correlations with most numerical columns
#Check its relation to loan_status
sns.countplot( x = 'total_acc',hue = 'loan_status',data = df_acc)

#Remove outlier data
df_total_acc = df_acc.groupby('total_acc').count()['loan_amnt'] == 1
df_total_acc['loan_amnt'] == True
sns.countplot( x = 'total_acc',hue = 'loan_status',data = df_acc[df_acc['total_acc'] > 125])## Check the plot for points above total acc 126
sns.heatmap(df_acc[df_acc['total_acc'] > 125].corr())
df_acc.drop(df_acc[df_acc['total_acc'] > 126].index, inplace = True)

del df_total_acc
del df_emp_len
# Fill mort_acc from total_acc column
df_mort_acc = df_acc.groupby('total_acc').mean()['mort_acc']
df_acc.groupby('mort_acc').describe()['total_acc']
def mort_acc_find(mort_acc_var,total_acc_var):
    if pd.isnull(mort_acc_var) == True:
        return int(df_mort_acc[total_acc_var])
    else:
        return mort_acc_var
df_acc['mort_acc_new'] = df_acc.apply(lambda x: mort_acc_find(x['mort_acc'],x['total_acc']),axis = 1)
df_acc['mort_acc_new'].unique()
df_acc['mort_acc'].value_counts()
df_acc.groupby('mort_acc_new').count()['mort_acc']
#Drop old Mort_acc
df_acc.drop('mort_acc',axis = 1 ,inplace = True)
del df_mort_acc

# Title
df_temp = df_acc.head(20) #To view sample data
df_acc.drop('title',axis = 1 ,inplace = True)

#revol_util
df_acc.corr()['revol_util'].sort_values(ascending = False)
df_acc['revol_util'].isnull().sum()#Only 857 compared to 1345287. Very less
sns.lmplot(x='revol_util',y='revol_bal',data = df_acc)

#Zip Code
df_acc['zip_code'].nunique()#943 columns to split for category is high
df_acc['addr_state'].nunique()
plt.figure(figsize=(12,4))
sns.countplot(x='addr_state',data = df_acc,hue= 'loan_status')#Not much to observe. CA has high loans and high fully paid too
#Drop zip code
df_acc.drop('zip_code',axis = 1,inplace = True)

#Now total missing values in whole dataframe is very less. We will drop them
df_acc.isnull().sum()
df_acc = df_acc.dropna()

#Now all categorical columns left
#Term   #Categorical
df_acc['term'].nunique()
df_acc.groupby('term').count()['loan_status']#60 months term they are more likely to charge off
sns.countplot(x='term',data = df_acc,hue= 'loan_status')#Not much to observe. CA has high loans and high fully paid too

#Grade and Sub Grade #Categorical
df_acc['sub_grade'].nunique()
plt.figure(figsize=(20,4))
sns.countplot(x='sub_grade',data = df_acc,hue= 'loan_status')#NF4 and G the charge off is almost equal to paid
df_acc['grade'].nunique()
df_acc.drop('grade',axis = 1 ,inplace = True)
df_acc.columns

#Home owners #Categorical
df_acc['home_ownership'].nunique()
sns.countplot(x='home_ownership',data = df_acc,hue= 'loan_status')#Not much info

#Verification Status #Categorical
df_acc['verification_status'].nunique()
sns.countplot(x='verification_status',data = df_acc,hue= 'loan_status')#Not much info

#Purpose #Categorical
df_acc['purpose'].nunique()
plt.figure(figsize=(20,4))
sns.countplot(x='purpose',data = df_acc,hue= 'loan_status')#Not much info

#Initial List status #Categorical
df_acc['initial_list_status'].nunique()
sns.countplot(x='initial_list_status',data = df_acc,hue= 'loan_status')#Not much info

#Application Type #Categorical
df_acc['application_type'].nunique()
sns.countplot(x='application_type',data = df_acc,hue= 'loan_status')#Not much info

#Convert loan_status to a 0/1 column(Already done)
#df_acc['loan_status'] = df_acc['loan_status'].apply(lambda x: 0 if (x == 'Charged Off') else  1)
#df_acc['loan_status'].unique()

#Numerical columns
sns.heatmap(df_acc.corr())
plt.figure(figsize=(20,4))
sns.pairplot(df_acc,hue='loan_status',palette='coolwarm')#Not a good idea
df_acc.corr()['loan_status'].sort_values().plot(kind='bar') # all columns have small +ve  or -ve correlation

#Fico columns have high correlation
df_acc['fico'] = (df_acc['fico_range_low'] + df_acc['fico_range_high'])/ 2
df_acc.drop('fico_range_low',axis = 1 ,inplace = True)
df_acc.drop('fico_range_high',axis = 1 ,inplace = True)
df_acc.groupby('loan_status').describe()['fico']
df_acc.groupby('loan_status').describe()['fico']

sns.lmplot(x = 'annual_inc',y = 'loan_amnt', data = df_acc, hue = 'loan_status' )
#Annual income is less charge off is high


#Date columns
df_acc['issue_d'].nunique()
df_acc['issue_d'] = df_acc['issue_d'].apply(pd.to_datetime)
df_acc['earliest_cr_line'] = df_acc['earliest_cr_line'].apply(pd.to_datetime)
plt.figure(figsize=(20,4))
sns.countplot(x='issue_d',hue='loan_status',data=df_acc)
plt.figure(figsize=(20,4))
sns.countplot(x='earliest_cr_line',hue='loan_status',data=df_acc)
df_acc.drop('issue_d',axis = 1 ,inplace = True) #not good to judge candidate before getting the loan
df_acc['cr_line_year'] = df_acc['earliest_cr_line'].dt.year
plt.figure(figsize=(20,4))
sns.countplot(x='cr_line_year',hue='loan_status',data=df_acc)
df_acc.drop('earliest_cr_line',axis = 1 ,inplace = True)

###Drop columns to see the rsult
df_acc.drop

#Apply algorithms
x = df_acc.drop('loan_status',axis =1)
df_temp = x.head(20)
y = df_acc['loan_status']
del df_temp
del x
x = pd.get_dummies(x,columns = ['term','sub_grade','home_ownership','verification_status','purpose','initial_list_status','application_type','addr_state'] ,drop_first = True)

#Split dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


#Scale data
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
x_train = x_scale.fit_transform(x_train)
x_test = x_scale.transform(x_test)

#from sklearn.preprocessing import MinMaxScaler
#x_scale = MinMaxScaler()
#x = x_scale.fit_transform(x)



#ANN on this
import keras 
from keras.models import Sequential
from keras.layers import Dense,Dropout

classifier = Sequential()
classifier.add(Dense(output_dim = 62, init = 'uniform',activation = 'relu',input_dim = 122))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 40, init = 'uniform',activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 20, init = 'uniform',activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',loss='binary_crossentropy')

classifier.fit(x_train, y_train, batch_size = 100, nb_epoch = 25,validation_data = (x_test,y_test))

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

#SVM
from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf',degree = 3,gamma = 'auto')
classifier1.fit(x_train,y_train)

# Predicting the Test set results
y_pred1 = classifier1.predict(x_test)
y_pred1 = (y_pred1 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)

#catBoost Classifier
from catboost import Pool, CatBoostClassifier
#cat_feat_ind = (x_train.dtypes == 'object').nonzero()[0]
cat_feat_ind = [1,4,5,7,8,9,16,17]
pool_train = Pool(x_train, y_train, cat_features=cat_feat_ind)
#pool_val = Pool(X_val, y_val, cat_features=cat_feat_ind)
pool_test = Pool(x_test, y_test, cat_features=cat_feat_ind)

n = y_train.value_counts()
model = CatBoostClassifier(learning_rate=.5,
                           iterations=350,
                           depth=3,
                           l2_leaf_reg=1,
                           random_strength=1,
                           bagging_temperature=1,
                           #grow_policy='Lossguide',
                           #min_data_in_leaf=1,
                           #max_leaves=1,
                           early_stopping_rounds=50,
                           class_weights=[1, n[0] / n[1]],
                           verbose=False,
                           random_state=0)
model.fit(pool_train, plot=True)

# Predicting the Test set results
y_pred1 = model.predict(x_test)
#y_pred1 = (y_pred1 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Get dataset
df_acc = pd.read_csv('accepted.csv')
df_acc = df_acc.iloc[:40000, :]
df_acc.head(10)

df_acc.columns

#Find columns that have maximum missing values
missing_fractions = df_acc.isnull().mean().sort_values(ascending=False)
df2 = (missing_fractions > 0.55) == True
df2 = pd.DataFrame(df2, columns = ['Binary1']).reset_index()
df2.groupby('Binary1').count()
df2[df2['Binary1'] == True]['index']
df_acc.drop(df2[df2['Binary1'] == True]['index'],axis = 1,inplace = True)

del missing_fractions

#Exploratory analysis
df_acc.info()
df_acc.describe()
df_acc.groupby('loan_status').count()
#Keep only Charged off and fully paid
df_temp = df_acc[(df_acc['loan_status'] == 'Charged Off') | (df_acc['loan_status'] == 'Fully Paid')]
df_acc = df_temp
#Columns to keep on basis of lending data study of best known columns for potential investors

keep_list = ['addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'grade', 'home_ownership', 'id', 'initial_list_status', 'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title', 'total_acc', 'verification_status', 'zip_code']
df_temp = df_acc
drop_list = [r for r in df_temp.columns if r not in keep_list]
df_temp = df_temp.drop(drop_list, axis = 1)
df_acc = df_temp


df_acc.columns
df_acc.isnull().mean().sort_values(ascending=False)
sns.boxplot(y = 'loan_amnt', x = 'loan_status',data = df_acc)
sns.boxplot(y = 'annual_inc', x = 'loan_status',data = df_acc)
df_acc.head(5)
df_corr = df_acc.corr()
sns.heatmap(df_corr)

#Remove remows with nan
df_temp.dropna(inplace = True)

#fico_high and low are correlated so add them
df_temp['fico'] = (df_temp['fico_range_high'] + df_temp['fico_range_low']) / 2
df_temp.drop('fico_range_high',axis = 1,inplace = True)
df_temp.drop('fico_range_low',axis = 1,inplace = True)
df_acc = df_temp
import re
df_temp['emp_length'].unique
df_temp['emp_length_new'] = df_temp['emp_length'].apply(lambda x: int(re.findall(r'\d+|$',str(x))[0]))
df_temp.drop('emp_length',axis = 1,inplace = True)

#Remove extra variables
del drop_list
del keep_list
del list1
del missing_fractions

#Identify categorical columns
df_temp['term'].unique() # Label encoding
df_temp['sub_grade'].unique() # Label encoding
df_temp['grade'].unique()
#Drop grade
df_temp.drop('grade',axis = 1,inplace = True)
df_acc = df_temp
df_temp['home_ownership'].unique()#Label Encoding
df_temp['verification_status'].unique() # Label Encoding
df_temp['purpose'].unique() # Label Encoding
df_temp['title'].unique() 
#Same as purpose mostly
df_temp.drop('title',axis = 1,inplace = True)
df_temp['zip_code'].unique()
#Too many to give conclusive results 
df_temp.drop('zip_code',axis = 1,inplace = True)
#Id is not needed as all unique
df_temp['id'].nunique()
df_temp.drop('id',axis = 1,inplace = True)

df_temp['addr_state'].unique() # Label Encoding
df_temp['emp_title'].unique() 
df_temp['emp_title'].nunique() #Too many values
df_temp.drop('emp_title',axis = 1,inplace = True)

df_temp['initial_list_status'].unique() # Label Encoding
df_temp['application_type'].unique()  # Label Encoding

#Relationship study of the categorical columns with o/p
sns.boxplot(x= 'term',y = 'loan_amnt', data = df_temp, hue = 'loan_status' )
# Loan amount for  60 months term is usually higher and charge off is also usually higher 

plt.figure(figsize =(12,5))
sns.factorplot(x ='sub_grade',y = 'loan_amnt', data = df_temp, hue = 'loan_status', kind = 'bar' )

#Joint application the charge off is higher
sns.factorplot(x ='application_type',y = 'loan_amnt', data = df_temp, hue = 'loan_status', kind = 'bar' )

#Verification status and initial_list_status  not much info
sns.factorplot(x ='initial_list_status',y = 'loan_amnt', data = df_temp, hue = 'loan_status', kind = 'bar' )
df_temp[['loan_amnt','loan_status','initial_list_status']].groupby('initial_list_status').count()

#Mortgage charge off is hidh and Any has no charge off(any is just 1 input)
sns.factorplot(x ='home_ownership',y = 'loan_amnt', data = df_temp, hue = 'loan_status', kind = 'bar' , estimator = np.mean)
df_temp[['loan_amnt','loan_status','home_ownership']].groupby('home_ownership').count()

#Verification status and charge off not much info
sns.factorplot(x ='verification_status',y = 'loan_amnt', data = df_temp, hue = 'loan_status', kind = 'bar' , estimator = np.mean)
df_temp[['loan_amnt','loan_status','verification_status']].groupby('verification_status').count()


#Explore Numerical Columns 
#Lower salaries have high charge off
sns.lmplot(x = 'annual_inc',y = 'loan_amnt', data = df_temp, hue = 'loan_status' )
list1 = df_temp.columns

df_temp[['loan_amnt','loan_status','pub_rec_bankruptcies']].groupby('pub_rec_bankruptcies').count()
df_temp[['loan_amnt','loan_status','pub_rec']].groupby('pub_rec').count()
sns.boxplot(x= 'pub_rec_bankruptcies',y = 'loan_amnt', data = df_temp, hue = 'loan_status' )
sns.boxplot(x= 'pub_rec',y = 'loan_amnt', data = df_temp, hue = 'loan_status' )


#Date columns to datetime--earliest_cr_line,issue_d
df_temp[['issue_d','earliest_cr_line']].head()
df_temp['issue_d'] = df_temp['issue_d'].apply(pd.to_datetime)
df_temp['earliest_cr_line'] = df_temp['earliest_cr_line'].apply(pd.to_datetime)
sns.factorplot(x ='issue_d',y = 'loan_amnt', data = df_temp, hue = 'loan_status', kind = 'bar' )
df_temp['issue_d_year'] = df_temp['issue_d'].dt.year
df_temp['issue_d_mnth'] = df_temp['issue_d'].dt.month
df_temp['issue_d_year'].nunique()

#Label Encoding and Onehot encoder
df_temp = pd.get_dummies(df_temp, columns = ['term','sub_grade','home_ownership','verification_status','purpose','initial_list_status','addr_state','application_type'], drop_first = True)
df_temp.shape
df_acc = df_temp

#Have X and Y
Y = df_temp['loan_status']
from sklearn.preprocessing import LabelEncoder
label_y = LabelEncoder()
Y = label_y.fit_transform(Y)
X = df_temp.drop('loan_status',axis = 1)
#Feature scaling
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
X[['loan_amnt','int_rate','installment','annual_inc','dti','open_acc','pub_rec','revol_bal','revol_util','total_acc','mort_acc','pub_rec_bankruptcies','fico','emp_length_new']] = x_scale.fit_transform(X[['loan_amnt','int_rate','installment','annual_inc','dti','open_acc','pub_rec','revol_bal','revol_util','total_acc','mort_acc','pub_rec_bankruptcies','fico','emp_length_new']])

#Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.15)

# Now Fit into a model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',degree = 3,gamma = 'auto')
classifier.fit(x_train,y_train)









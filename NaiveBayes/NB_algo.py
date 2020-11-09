# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# # NAIVE BAYES
# %% [markdown]
# # LOAD THE DEPENDANCIES
# %% [markdown]
# ## Pandas
# 

# %%
import pandas as pd
from pandas import set_option
from pandas.plotting import scatter_matrix

# %% [markdown]
# ## Numpy

# %%
import numpy as np
from numpy import set_printoptions

# %% [markdown]
# ## Matplotlib & Seaborn

# %%

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set()
# import graphviz

# %% [markdown]
# ## sklearn

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA 
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## Math & statistics

# %%
from scipy import stats
from scipy.stats import norm
import math

# %% [markdown]
# ## System

# %%
import os
import sys
import pprint
# sys.path.insert(0, "C:\\Users\\Crystal\\Desktop\\Programs\\my-modules-and-libraries")
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# %% [markdown]
# ## notebook widgets

# %%
# import ipywidgets as widgets
from IPython.display import Image
from IPython.display import display, Math, Latex
from IPython.core.interactiveshell import InteractiveShell  
InteractiveShell.ast_node_interactivity = "all"

# %% [markdown]
# # FUNCTIONS
# %% [markdown]
# ## Label Encoding

# %%
def label_encoding(dataset,input_headers):
    
    for i in input_headers:
        
        the_data_type=dataset[i].dtype.name
        if (the_data_type=='object'):
            lable_enc=preprocessing.LabelEncoder()
            lable_enc.fit(dataset[i])
            labels=lable_enc.classes_   #this is an array
            labels=list(labels) #converting the labels array to a list
            print(labels)
            dataset[i]=lable_enc.transform(dataset[i])

            return labels
    
        else:
            c=list(np.unique(dataset[i]))
            return [str(x) for x in c]

# %% [markdown]
# ## Feature Scaling

# %%
def feature_scaling(X_train,X_test):
    sc_X=StandardScaler()
    X_train=sc_X.fit_transform(X=X_train,y=None)
    X_test=sc_X.fit_transform(X=X_test,y=None)

    print(sc_X.fit(X_train))
    print(X_train[0:5])
    
    
    
    return X_train, X_test

# %% [markdown]
# ## Visualization
# %% [markdown]
# ### Plot the data space (scatter)

# %%
def plot_of_data_space(dataset,data,labels,input_headers):
    
    
    xx_1=pd.DataFrame(data[:,0]) 
    xx_2=pd.DataFrame(data[:,1]) 
    y=pd.DataFrame(labels)
    
   
    plt.figure(figsize=(15,10)) 
    b=plt.scatter(xx_1[y==0],xx_2[y==0],color='b') 
    r=plt.scatter(xx_1[y==1],xx_2[y==1],color='r')
    g=plt.scatter(xx_1[y==2],xx_2[y==2],color='g') 
    bl=plt.scatter(xx_1[y==3],xx_2[y==3],color='black')
    
    
#     for i in range(0,len(xx_1)):
#         print(y[i])
#         if (y[i]==0):
#             a=plt.scatter(xx_1[i],xx_2[i],marker='o',color='blue',s=30)
#         if (y[i]==1):
#             b=plt.scatter(xx_1[i],xx_2[i],marker='o',color='red',s=30)
#         if (y[i]==2):
#             c=plt.scatter(xx_1[i],xx_2[i],marker='o',color='green',s=30)
#         if (y[i]==3):
#             d=plt.scatter(xx_1[i],xx_2[i],marker='o',color='black',s=30)
        
#     plt.xlabel(f1);plt.ylabel(f2);
#     plt.legend((a,b),tuple(np.unique(labels)))

    plt.xlabel(input_headers[0])
    plt.ylabel(input_headers[1])

    plt.grid()
    plt.legend((b,r,g,bl),tuple(np.unique(labels)))
    plt.show()

# %% [markdown]
# ### Feature Distributions (histograms)

# %%
def feature_distributions(df,target_header,*args):
    
    
    data=df.drop(target_header,axis=1,inplace=False)

    num_plot_rows=len(data.columns)

    print (classes)
    
    label_encoder = preprocessing.LabelEncoder()
    df[target_header]=label_encoder.fit_transform(df[target_header])
    labels=label_encoder.classes_   #this is an array
    labels=list(labels) #converting the labels array to a list
    print (labels)

    fig = plt.figure(figsize = (20,num_plot_rows*4))
    j = 0

    ax=[]
    colors=['b','r','g','black']
    for i in data.columns:
        plt.subplot(num_plot_rows, 4, j+1)
        j += 1
        for k in range(len(labels)):
    #         print(k)
            a=sns.distplot(data[i][df[target_header]==k], color=colors[k], label = str(labels[k])+classes[k]);
            ax.append(a)
        plt.legend(loc='best')
    
    fig.suptitle(target_header+ ' Data Analysis')
    fig.tight_layout()
    # fig.subplots_adjust(top=0.95)
    plt.show()

# %% [markdown]
# ## Preprocessing: Splitting the dataset

# %%


def split_the_dataset(dataset,input_headers,target_header):
    
    X=dataset[input_headers]
    y=dataset[target_header]
    
    X.head()
    
    return X,y



# %% [markdown]
# ## Replacing Zeros

# %%
def replacing_zeros(dataset,the_headers):
    """Function used to remove zeros from numeric features when 0 is not practical"""

    for header in the_headers:
        dataset[header]=dataset[header].replace(0,np.nan)
        mean=int(dataset[header].mean(skipna=True))
        dataset[header]=dataset[header].replace(np.nan,mean)
        
    return dataset

# %% [markdown]
# ## Feature Correlations

# %%
def correlation_matrix(dataset,input_headers,target_header):
    
    feature_matrix=dataset[input_headers]
    corr=feature_matrix.corr()
    corr
    
    plt.figure(figsize=(10,10))
    corr_plot=sns.heatmap(corr,cmap="Reds",annot=True)
    
    corr_pair=sns.pairplot(dataset,hue=target_header[0])
    plt.show()
    
    return corr,corr_plot,corr_pair 
   

# %% [markdown]
# ## Drop Unwanted Features

# %%
def feature_drop(dataset,headers_to_drop):
    
    dataset.drop(labels=headers_to_drop,axis=1,inplace=True)
    dataset.head()

# %% [markdown]
# ## Principal Component Analysis (PCA)

# %%


def pca(dataset,input_headers,target_header,*args):
    
    feature_matrix=dataset[input_headers]
    model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
    model.fit(feature_matrix)  # 3. Fit to data. Notice y is not specified!
    X_2D = model.transform(feature_matrix)         # 4. Transform the data to two dimensions


    dataset['PCA1'] = X_2D[:, 0]
    dataset['PCA2'] = X_2D[:, 1]

    sns.lmplot("PCA1", "PCA2", hue=target_header[0], data=dataset, fit_reg=False);
    
    
#     sns.distplot(dataset['PCA1'][dataset[target_header[0]]==0], color='b', label = '0')
#     sns.distplot(dataset['PCA1'][dataset[target_header[0]]==1], color='r', label = '1')
#     # sns.distplot(df['PCA1'][df[target_header]==2], color='r', label = '2')
#     plt.legend(loc='best')
#     plt.show()

# %% [markdown]
# # MAIN PROGRAM
# %% [markdown]
# ## Get Data

# %%
if __name__ == "__main__":
    
    location=r'C:\Users\Crystal\Desktop\Programs\dataset_repo\LCDS Datasets\heart_failure_clinical_records_dataset.csv'
#     location=r'C:\Users\Crystal\Desktop\Programs\dataset_repo\CDH_Train.csv'
#     location=r'C:/Users/Crystal/Desktop/Programs/dataset_repo/0529_/0529_pass_rush.csv'
    dataset=pd.read_csv(location)
    # df=pd.read_csv('thermostat_dataset.txt',delimiter='\t')

    dataset.info()
    dataset.head() 
    dataset.describe()

# %% [markdown]
# ## Drop unwanted features (columns)

# %%
all_cols=list(dataset.columns)
all_cols


# %%

the_target=['DEATH_EVENT']
selected_cols=['age',
 'anaemia',
 'platelets',
 'serum_sodium',the_target[0]]


# %%
drop_these=list(set(all_cols).difference(set(selected_cols)))
drop_these


# %%
drop_columns=drop_these
if (drop_columns!=[]):
    q1=input('Do you need to drop any columns in the dataset?')
    if (q1.lower()=='y'):
        feature_drop(dataset,drop_columns)


# %%
dataset.head()

# %% [markdown]
# ## Selecting inputs and targets

# %%

target_header=the_target
selected_cols.remove(target_header[0])
input_headers=selected_cols
print(target_header)
print(input_headers)
target_label=label_encoding(dataset,target_header)

classes=target_label
print (classes)
test_label=label_encoding(dataset,input_headers)

dataset=dataset[input_headers+target_header]
X,y=split_the_dataset(dataset,input_headers,target_header)

print(X.head())

# %% [markdown]
# ## Replace zeros with the mean where needed.

# %%
rz=input('Do you need to replace any zeros in the dataset?')
if (rz.lower()=='y'):
    the_headers=b
    dataset=replacing_zeros(dataset,the_headers)
    dataset.head()

# %% [markdown]
# ## Data Visualizations
# %% [markdown]
# ### Data space

# %%
if (X.values.shape[1]==2):
    plot_of_data_space(dataset,X.values,y.values,input_headers)
else:
    pca(dataset,input_headers,target_header)


# %%
target_header[0]

# %% [markdown]
# ### Feature distributions

# %%
dataset.head()


# %%
feature_distributions(dataset,target_header[0],classes)


# %%
X.head()

# %% [markdown]
# ## Correlation Matrix

# %%
correlation_matrix(dataset,input_headers,target_header)


# %%
y.head()

# %% [markdown]
# ## Splitting the Train-Test data

# %%
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.20)


# %%
ytest.shape

# %% [markdown]
# ## Scale the data

# %%
#Scale the data    
Xtrain, Xtest=feature_scaling(Xtrain,Xtest)


# %%
ytest.head()

# %% [markdown]
# ## Naive Bayes Model

# %%
model = GaussianNB(priors=None)

# %% [markdown]
# ## Fit model to training data

# %%
model.fit(Xtrain,ytrain)

# %% [markdown]
# ## Model prediction on test data

# %%
y_model=model.predict(Xtest)
y_model

# %% [markdown]
# ## Model score & performance

# %%
accuracy_score(ytest,y_model)


# %%
recall_score(ytest, y_model,average=None)


# %%
precision_score(ytest, y_model,average=None)

# %% [markdown]
# ### Confusion Matrix

# %%
cm=confusion_matrix(ytest, y_model)


# %%
cm


# %%
fig, ax = plt.subplots()
cmap=plt.cm.binary
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes,
       yticklabels=classes,
       title="confusion",
       ylabel='True label',
       xlabel='Predicted label')



# Loop over data dimensions and create text annotations.
normalize=False
fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

plt.grid(False)

plt.show()


# %%
sns.heatmap(cm,square=True,annot=True,cbar=True)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

# %% [markdown]
# ### Cross Validation

# %%
y=y.values.reshape(y.size,)
score=cross_val_score(model,X,y,cv=10)


# %%
score


# %%
score.mean()


# %%
sns.boxplot(x=score,orient='v')
plt.show()





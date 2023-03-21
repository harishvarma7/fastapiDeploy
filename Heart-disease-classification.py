#!/usr/bin/env python
#coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 



# we want our plots to appear inside notebook
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import plot_roc_curve


# ## Load Data

# In[2]:


df=pd.read_csv("heart-disease.csv")
df.shape


# In[ ]:


df["target"].value_counts()


# In[ ]:


df["target"].value_counts().plot(kind="bar", color=["salmon", "lightblue"])


# In[ ]:


df.info()


# In[ ]:


#compare target column with sex column

pd.crosstab(df.target, df.sex)


# In[ ]:


pd.crosstab(df.target, df.sex).plot(kind="bar",figsize=(10,6), color=["salmon","lightblue"])

plt.title("Heart Disease frequency for sex")
plt.xlabel("0= No Disease, 1=Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"])
plt.xticks(rotation=0);
# To align 0 and 1 vertically in the x axis


# ## Age vs Max heart rate for heart disease

# In[ ]:


plt.figure(figsize=(10,6))

plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="salmon");

# scatter with negative examples

plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="lightblue");

plt.title("Heart Disease in function of Age and max heart Rate")
plt.xlabel("Age")
plt.ylabel("max heart rate")
plt.legend(["Disease","No Disease"])


# In[ ]:


# check the distribution of age column with a hsitogram

df.age.plot.hist()


# In[ ]:


#Correlation matrix
corr_matrix=df.corr()
fig, ax=plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_matrix,annot=True,
               linewidths=0.5,
               fmt=".2f");
bottom,top=ax.get_ylim()
ax.set_ylim(bottom + 0.5, top-0.5)


# In[ ]:


# split data into x and y
x=df.drop("target", axis=1)
y=df["target"]
#split into train and test sets
np.random.seed(0)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# we will try three machine learning models
# 1.Logistic Regression
# 2.K-Nearest neighbour classifier
# 3.Random forest Classifier
# 

# In[ ]:


#put models in a dictionary
models={"Logistic Regression": LogisticRegression(),
       "KNN":KNeighborsClassifier(),
       "random Forest": RandomForestClassifier()}

#Create a function to fit and score models

def fit_and_score(models,x_train,x_test,y_train,y_test):
    """
     Fits and evaluates given machine learning models.
     models: a dict of different Scikit-learn machine learning models
    """
    np.random.seed(0)
    #make a dictionary to keep model scores
    model_scores={}
    #loop through models
    for name, model in models.items():
        #Fit the model to the data
        model.fit(x_train,y_train)
        #Evaluate the model
        model_scores[name]=model.score(x_test,y_test)
    return model_scores
    


# In[ ]:


model_scores=fit_and_score(models=models,x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test)
model_scores


# ## Model comparison
# 

# In[ ]:


model_compare=pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar()


# ### Hyper parameter Tuning
# 
# 

# In[ ]:



train_scores=[]
test_scores=[]

neighbors=range(1,21)

knn=KNeighborsClassifier()

for i in neighbors:
    knn.set_params(n_neighbors=i)
    knn.fit(x_train,y_train)

    train_scores.append(knn.score(x_train,y_train))

    test_scores.append(knn.score(x_test,y_test))


# In[ ]:


train_scores


# In[ ]:


test_scores


# In[ ]:


plt.plot(neighbors,train_scores,label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# ## Hyper parameter tuning using Randomized search cv
# we are going to tune:
# * Logistic regression()
# * RandomForestClassifier()

# In[ ]:


#Create a hyperparameter grid for LogisticRegression
log_reg_grid={"C": np.logspace(-4,4,20),
             "solver":["liblinear"]
             }
#Create a hyperparameter grid for RandomForestClassifier
rf_grid={"n_estimators":np.arange(10,1000,50),
         "max_depth":[None,3,5,10],
         "min_samples_split":np.arange(2,20,2),
         "min_samples_leaf":np.arange(1,20,2)
    
    
}


# In[ ]:


np.random.seed(0)

rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                             param_distributions=log_reg_grid,
                              cv=5,
                              n_iter=20,
                              verbose=True
                             )

rs_log_reg.fit(x_train,y_train)


# In[ ]:


rs_log_reg.best_params_


# In[ ]:


rs_log_reg.score(x_test,y_test)


# In[ ]:


np.random.seed(7)
rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                        param_distributions=rf_grid,
                        cv=5,
                         n_iter=20,
                         verbose=True
                        )

rs_rf.fit(x_train,y_train)


# In[ ]:


rs_rf.best_params_


# In[ ]:


rs_rf.score(x_test,y_test)


# ## Hyper parameter tuning GridSearchCV
# 

# In[ ]:


log_reg_grid={"C": np.logspace(-4,4,30),
              "solver": ["liblinear"]}

             
gs_log_reg=GridSearchCV(LogisticRegression(),
                           param_grid=log_reg_grid,
                           cv=5,
                           verbose=True)
gs_log_reg.fit(x_train,y_train)


# In[ ]:


gs_log_reg.best_params_


# In[ ]:


gs_log_reg.score(x_test,y_test)


# ### Evaluating our tuned machine learning classifier

# In[ ]:


y_preds=gs_log_reg.predict(x_test)
y_preds


# In[ ]:


plot_roc_curve(gs_log_reg,x_test,y_test)


# In[ ]:


print(classification_report(y_test,y_preds))


# In[ ]:


gs_log_reg.best_params_


# In[ ]:


#Create a new classifier with best parameters
clf=LogisticRegression(C= 1.3738237958832638,
                      solver="liblinear")


# In[ ]:



cv_acc=cross_val_score(clf,x,y,
                      cv=5,
                      scoring="accuracy")
cv_acc


# In[ ]:


np.mean(cv_acc)


# In[ ]:


cv_precision=cross_val_score(clf,x,y,
                      cv=5,
                      scoring="precision")
cv_precision=np.mean(cv_precision)
cv_precision


# In[ ]:


cv_recall=cross_val_score(clf,x,y,
                      cv=5,
                      scoring="recall")
cv_recall=np.mean(cv_recall)
cv_recall


# In[ ]:


cv_f1=cross_val_score(clf,x,y,
                      cv=5,
                      scoring="f1")
cv_f1=np.mean(cv_f1)
cv_f1


# In[ ]:


#Visualize cross validated metrics
cv_metrics=pd.DataFrame({"Accuracy":cv_acc,
                        "Precision":cv_precision,
                        "Recall":cv_recall,
                        "F1": cv_f1},
                         
                       )
cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                     legend=False);


# ## Feature importance
# Feature importance is another as asking, "Which features contributed most to theoutcomes of the model and how did they contribute?"

# In[ ]:


df.head()


# In[ ]:


gs_log_reg.best_params_

clf=LogisticRegression(C=1.3738237958832638,
                      solver="liblinear")

clf.fit(x_train,y_train)


# In[ ]:


clf.coef_


# In[ ]:


#Match coef's of features to columns
feature_dict=dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[ ]:


#Visualize feature importance
feature_df=pd.DataFrame(feature_dict,index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);


# In[ ]:
pickle.dump(clf,open("model.pkl","wb"))
print("succesfully exported model")





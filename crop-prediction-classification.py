

from __future__ import print_function

import numpy
import pandas as pd # data analysis
import numpy as np # linear algebra

#import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')





crop = pd.read_csv('Data/crop_recommendation.csv')
crop.head(5)





crop.info()


# In[4]:


crop.describe()


# In[5]:


crop.columns


# In[6]:


crop.shape


# In[7]:


crop['label'].unique()


# In[8]:


crop['label'].nunique()


# In[9]:


crop['label'].value_counts()


# In[10]:


sns.heatmap(crop.isnull(),cmap="coolwarm")



# In[11]:


plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)

sns.distplot(crop['temperature'],color="red",bins=15,hist_kws={'alpha':0.5})
plt.subplot(1, 2, 2)
sns.distplot(crop['ph'],color="green",bins=15,hist_kws={'alpha':0.5})


# In[12]:


sns.pairplot(crop,hue = 'label')


# In[13]:


sns.jointplot(x="rainfall",y="humidity",data=crop[(crop['temperature']<40) & 
                                                  (crop['rainfall']>40)],height=10,hue="label")


# In[14]:


sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(30,15))
sns.boxplot(x='label',y='ph',data=crop)


# In[15]:


fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(crop.corr(), annot=True,cmap='viridis')
ax.set(xlabel='features')
ax.set(ylabel='features')

plt.title('Correlation between different features', fontsize = 15, c='black')



# In[16]:


crop_summary = pd.pivot_table(crop,index=['label'],aggfunc='mean')
crop_summary.head()


# In[17]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['N'],
    name='Nitrogen',
    marker_color='mediumvioletred'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['P'],
    name='Phosphorous',
    marker_color='springgreen'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['K'],
    name='Potash',
    marker_color='dodgerblue'
))

fig.update_layout(title="N-P-K values comparision between crops",
                  plot_bgcolor='white',
                  barmode='group',
                  xaxis_tickangle=-45)







features = crop[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = crop['label']


# In[19]:


acc = []
model = []


# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features,target,test_size = 0.2,random_state =2)




from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DT.fit(x_train,y_train)

predicted_values = DT.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("Decision Tree's Accuracy is: ", x*100)

print(classification_report(y_test,predicted_values))
dtacc=metrics.accuracy_score(y_test, predicted_values)
dtf1= metrics.f1_score(y_test, predicted_values,average='macro')
dtRecall=metrics.recall_score(y_test, predicted_values,average='macro')
dtPrecision=metrics.precision_score(y_test, predicted_values,average='macro')

# In[36]:


score = cross_val_score(DT, features, target,cv=5)
print('Cross validation score: ',score)


# In[37]:


#Print Train Accuracy
dt_train_accuracy = DT.score(x_train,y_train)
print("Training accuracy = ",DT.score(x_train,y_train))
#Print Test Accuracy
dt_test_accuracy = DT.score(x_test,y_test)
print("Testing accuracy = ",DT.score(x_test,y_test))


# In[38]:


y_pred = DT.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_dt = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(15,10))
sns.heatmap(cm_dt, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')






from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(x_train,y_train)

predicted_values = RF.predict(x_test)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('RF')
print("Random Forest Accuracy is: ", x)

print(classification_report(y_test,predicted_values))
rfacc=metrics.accuracy_score(y_test, predicted_values)
rff1= metrics.f1_score(y_test, predicted_values,average='macro')
rfRecall=metrics.recall_score(y_test, predicted_values,average='macro')
rfPrecision=metrics.precision_score(y_test, predicted_values,average='macro')

# In[40]:


score = cross_val_score(RF,features,target,cv=5)
print('Cross validation score: ',score)


# In[41]:


#Print Train Accuracy
rf_train_accuracy = RF.score(x_train,y_train)
print("Training accuracy = ",RF.score(x_train,y_train))
#Print Test Accuracy
rf_test_accuracy = RF.score(x_test,y_test)
print("Testing accuracy = ",RF.score(x_test,y_test))


# In[42]:


y_pred = RF.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(15,10))
sns.heatmap(cm_rf, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')




N = 2
ind = np.arange(N)  # the x locations for the groups
width = 0.1       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [dtacc,rfacc]
rects1 = ax.bar(ind, yvals, width, color='r')
zvals = [dtf1,rff1]
rects2 = ax.bar(ind+width, zvals, width, color='g')
kvals = [dtRecall,rfRecall]
rects3 = ax.bar(ind+width*2, kvals, width, color='b')
mvals = [dtPrecision,rfPrecision]
rects4 = ax.bar(ind+width*3, mvals, width, color='y')

ax.set_ylabel('Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Decision Tree','Random Forest') )
ax.legend((rects1[0], rects2[0], rects3[0],rects4[0]), ('Accuracy', 'F1-Measure', 'Recall','Precision') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 2.05*h, '%d'%int(h),ha='center', va='bottom')


plt.savefig('static/images/plot1.png', dpi=500, bbox_inches='tight')









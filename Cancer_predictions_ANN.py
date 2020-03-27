#!/usr/bin/env python
# coding: utf-8

# In[131]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[132]:


import os


# In[133]:


os.chdir("D:/New Projects/Tensor/TensorFlow_FILES/DATA")


# In[134]:


df = pd.read_csv("cancer_classification.csv")


# In[135]:


df.head()


# In[136]:


df.info()


# In[137]:


df.describe().transpose()


# In[138]:


sns.countplot(x='benign_0__mal_1',data=df)


# In[139]:


plt.figure(figsize=(10,6))
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')


# In[140]:


sns.heatmap(df.corr())


# In[141]:


X=df.drop('benign_0__mal_1',axis=1).values


# In[142]:


y=df['benign_0__mal_1'].values


# In[143]:


from sklearn.model_selection import train_test_split


# In[144]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[145]:


from sklearn.preprocessing import MinMaxScaler


# In[146]:


scaler=MinMaxScaler()


# In[147]:


X_train=scaler.fit_transform(X_train)


# In[148]:


X_test=scaler.transform(X_test)


# In[149]:


from tensorflow.keras.models import Sequential


# In[150]:


from tensorflow.keras.layers import Dense,Dropout


# In[151]:


X_train.shape


# In[152]:


model= Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')


# In[153]:


model.fit(x=X_train,y=y_train,epochs=600, validation_data=(X_test,y_test))


# In[154]:


losses=pd.DataFrame(model.history.history)


# In[155]:


losses.plot()


# In[156]:


model= Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')


# In[157]:


from tensorflow.keras.callbacks import EarlyStopping


# In[158]:


help(EarlyStopping)


# In[159]:


early_stop=EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=25)


# In[160]:


model.fit(x=X_train,y=y_train,epochs=600, validation_data=(X_test,y_test), callbacks=[early_stop])


# In[161]:


model_losses=pd.DataFrame(model.history.history)


# In[162]:


model_losses.plot()


# In[163]:


model= Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')


# In[164]:


model.fit(x=X_train,y=y_train,epochs=600, validation_data=(X_test,y_test), callbacks=[early_stop])


# In[165]:


model_losses_dropout=pd.DataFrame(model.history.history)


# In[166]:


model_losses_dropout.plot()


# In[167]:


predictions=model.predict_classes(X_test)


# In[168]:


from sklearn.metrics import classification_report, confusion_matrix


# In[169]:


print(classification_report(y_test,predictions))


# In[170]:


print(confusion_matrix(y_test,predictions))


# In[ ]:





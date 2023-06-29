import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import  Adam
from keras.regularizers import l2
from keras.activations import relu,linear
from keras.losses import SparseCategoricalCrossentropy

data=pd.read_csv('train.csv')
#print(data.shape)
np_array=data.values
np_array=np_array.astype(float)
y_train=np_array[:,0]
y_train=np.reshape(y_train,(-1,1))

np_array=np.delete(np_array,[0],axis=1)
#print(np_array.dtype)
#print(np_array.shape)

#this section was to generate cross validation set
'''v80=int(np_array.shape[0]*.8)
x_cv=np_array[v80:]
x_train=np_array[:v80]'''
#print(x_train[:5])

#print(x_train.shape)

x_train=np_array
mean=np.average(x_train,axis=0)
#print(mean)
std=np.std(x_train,axis=0)
#print(std)
x_train=(x_train-mean)/(std+(1e-12))
#print((x_train[0]))

#for cv set
'''y_cv=y_train[v80:]
y_train=y_train[:v80]'''

#computed the error in cross validation set for different values of regularization and found that 0 regularization is best
#reason is that already a large no of examples are present so overfitting does not occur

model=Sequential([
    Dense(units=112,activation=relu,kernel_regularizer=l2(0)),
    Dense(units=56,activation=relu,kernel_regularizer=l2(0)),
    Dense(units=28,activation=relu,kernel_regularizer=l2(0)),
    Dense(units=10,activation=linear,kernel_regularizer=l2(0))
])

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(.001)
)

model.fit(x_train,y_train,epochs=15)

'''x_cv=(x_cv-mean)/(std+(1e-12))
y_cv_output=model.predict(x_cv)
y_output=[]
for i in y_cv_output:
    index=np.argmax(i)
    y_output.append(index)
print(y_output)

print(y_cv.shape)
count=0
for i in range(y_cv.shape[0]):
    if y_cv[i][0]!=y_output[i]:
        count+=1

print(f'{count}/{y_cv.shape[0]}')'''


test_set=pd.read_csv('test.csv')
x_test=test_set.values

x_test=x_test.astype(float)
x_test=(x_test-mean)/(std+(1e-12))
y_test=model.predict(x_test)

y_output=[]
for i in y_test:
    index=np.argmax(i)
    y_output.append(index)
print(y_output)

y_output=np.array(y_output).reshape((-1,1))
imageId=np.arange(1,x_test.shape[0]+1).reshape((-1,1))
imageId=imageId.astype(int)
y_output=y_output.astype(int)

headers="ImageId,Label"
output_file="output.csv"
np.savetxt(output_file,np.concatenate([imageId,y_output],axis=1),delimiter=',',header=headers,comments="",fmt='%d')

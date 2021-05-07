import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten, Conv2D,MaxPooling2D
from tensorflow.keras.utils import normalize
import numpy as np
import os
import cv2
import random
import pickle
categories=['nude','sexy','safe']
directory='D:\\NudeNet_classifier_dataset_v1\\NudeNet_Classifier_train_data_x320\\nude_sexy_safe_v1_x320\\training'
xpath=os.path.join(directory,"x.pickle")
ypath=os.path.join(directory,"y.pickle")
training_data=[]
x=[]
y=[]
img_size=256
if(os.path.isfile(xpath) and os.path.isfile(ypath)):
    pickle_in=open(xpath,"rb")
    x=pickle.load(pickle_in)
    pickle_in.close()
    pickle_in=open(ypath,"rb")
    y=pickle.load(pickle_in)
    pickle_in.close()
else:
    count=0
    for c in categories:
        path=os.path.join(directory,c)
        print(c)
        for img in os.listdir(path):
            print(img)
            try:
                img_data=cv2.imread(os.path.join(path,img))
                new_data=cv2.resize(img_data,(img_size,img_size))
                training_data.append([new_data,categories.index(c)])
                count+=1
            except:
                print('Failed'+img)
            if(count>1000):
                count=0
                break
 
 
    random.shuffle(training_data)
    print(training_data)


    for i,j in training_data:
        print(i,j)
        
        x.append(i)
        y.append(j)

    #x=np.array(x).reshape(-1,1)


    pickle_out=open(xpath,"wb")
    pickle.dump(x,pickle_out)
    pickle_out.close()

    pickle_out=open(ypath,"wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()


x=np.array(x).reshape(-1,img_size,img_size,3)
y=np.array(y)

x=x/255.0
#x=normalize(x,axis=-1)
print(x.shape)
#print(y.shape)
model= Sequential()
model.add(Conv2D(64,(3,3), input_shape=(img_size,img_size,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(x,y,epochs=10,batch_size=64)

#model.save('nsfw_classifier'),batch_size=64







        
        

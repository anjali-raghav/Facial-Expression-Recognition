#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import cv2
images=[]
for file in glob.glob("F:\Project Report\emotion detection\CK+48\anger\*.png"):
    img=cv2.imread(str(file))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    images.append(img)

images

images.insert(0,'P')

# images.remove('Pixels')

import csv
with open('testck.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in range(len(images)):
        wr.writerow((0,list(images[i])))



import pandas as pd
ff= pd.read_csv('testck.csv')

ff

import numpy as np
for i in range(len(images)):
    images[i]=np.array(images[i]).flatten()
#     print(len(i))

images[0].shape

h=[]
for file in glob.glob(r"F:\Project Report\emotion detection\CK+48\contempt\*.png"):
    img=cv2.imread(str(file))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    h.append(img)

for i in range(len(h)):
    h[i]=np.array(h[i]).flatten()


# In[2]:


#opening the csv in append mode
path="F:/Project Report/emotion detection"
with open('testck.csv','a', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in range(len(h)):
        wr.writerow((6,list(h[i])))

ff= pd.read_csv('testck.csv')
ff.to_csv(path+ '/'  + "testck.csv" ,index=False)


# In[3]:


ff


# In[5]:



test_file=pd.read_csv(path + '/' +'testck.csv')


# In[6]:


import os
path="F:\Project Report\emotion detection"
for i in os.listdir(path):
    print(i)


# In[7]:


file=pd.read_csv(path+ '/' + "fer2013.csv")


# In[8]:


print('file=',file)
print('test_file=',test_file)


# In[9]:


#code to fetch all the data from imported csv files
#x and y
X=[]
Y=[]
for i in range(file.shape[0]):
  l=file.iloc[i,1].split()
  l=[float(ele) for ele in l] 
#   print('len=',len(l))
  print(i)
#   if file.iloc[i,2] == "Training":
  X.append(np.array(l).reshape(48,48))
  Y.append(file.iloc[i,0])


# In[10]:


#printing the data X and Y and corresponding shape
X=np.asarray(X)
Y=np.asarray(Y)


# In[11]:



print('X.shape=',X.shape)
print('Y.shape=',Y.shape)


# In[12]:


# normalizing the data

def normalize(x):
     # Normalize inputs to (0,1)
#     print(x.shape)
    print(x[0],x[1],x[2])
    x_n = x/255.
    return x_n
X = normalize(X)


# In[13]:


#resahape image-pixels  inputs to classify to grayscale (3 ,if rgb)
def reshape(x):
    x_r=x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    print(x_r.shape)
    return x_r
X = reshape(X)


# In[14]:


#one hot encoding
def oneHot(y, Ny):
    
    from keras.utils import to_categorical 
    y_oh=to_categorical(y,num_classes=Ny)
    return y_oh

# example
Y = oneHot(Y,7)


# In[15]:


# splitting the whole data to train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.1,random_state=1)


# In[16]:


x_train.shape


# In[ ]:


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train.argmax(1)), y_train.argmax(1))


# In[ ]:


from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2D, AveragePooling2D
model = Sequential()
#1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
 
#2nd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
 
model.add(Flatten())
 
#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
 
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

model.summary()


# In[1]:


from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="F:/Project Report/emotion detection/best_model_weights.hdf5", verbose=1, save_best_only=True)
# model.load_weights('weights.hdf5')
history = model.fit(x_train, y_train, validation_split=0.1,epochs=25, batch_size=200, callbacks=[checkpointer], class_weight=class_weights)


# In[25]:


import h5py


# In[1]:


import os
from keras.models import load_model
path2="F:\Project Report\emotion detection"
loaded_model=load_model(path2 +'/'+'best_model_weights.hdf5')


# In[2]:


loaded_model.summary()


# In[19]:


#output prediction
def predict(x):
    
    y=loaded_model.predict(x)
    return y


# In[20]:


def oneHot_tolabel(y):
    
    y_b = np.argmax(y,axis=1)
    np.asarray(y_b)
    return y_b
# oneHot_tolabel(train_labels)

y_pred=loaded_model.predict(x_train)


# In[21]:


def create_confusion_matrix(true_labels, predicted_labels):
    
#     print('predicted_labels=',predicted_labels.shape)
#     print()

    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=1))
    return cm


# In[22]:


cm = create_confusion_matrix(oneHot_tolabel(y_test), oneHot_tolabel(predict(x_test)))

print("Confusion Matrix")
print(cm)


# In[24]:


from sklearn.metrics import classification_report
print(classification_report(oneHot_tolabel(y_test), oneHot_tolabel(predict(x_test))))  


# In[30]:


import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[31]:


plot_confusion_matrix(cm, normalize    = True, target_names = ['Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral'],
                      title        = "Confusion Matrix-Normalised")


# In[ ]:


history.history.keys()
import matplotlib.pyplot as plt
plt.plot(range(len(history.history['val_acc'])), history.history['val_acc'])
plt.show()


# In[24]:


def accuracy(x_test, y_test, model):
    
    loss,acc=model.evaluate(x_test,y_test,verbose=0)
    return acc

# acc = accuracy(x_test_fer, y_test_fer, model)

acc = accuracy(x_train, y_train, loaded_model)
print('Training accuracy is, ', acc*100, '%')


# In[25]:


wrong=list(np.nonzero(loaded_model.predict(x_test).argmax(1)!= y_test.argmax(1)))[0]
# print('y_test.argmax(1)=',y_test.argmax(1))
# type(wrong)
# print('wrong=',wrong)
# print('wrong=',len(wrong))

emotion_ls=['Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']
# anger=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6

emotion_ls=['Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']
import numpy as np
correct=[20,2,46]
index=np.arange(7)
# emotions=pd.file.iloc[]
y_pred=loaded_model.predict (x_test)


# In[ ]:


def plot_img(i):

  import matplotlib.pyplot as plt

  x=x_test[i]
  x=x.reshape(48,48)
  pixels=x
  plt.imshow(pixels,cmap='gray')
  plt.show()
  print('label of img is ',emotion_ls[y_test[i].argmax()])
  print('pred of img is ',emotion_ls[y_pred[i].argmax()])
  print('y_pred=',y_pred[i])
  plt.bar(index, y_pred[i])
  plt.xlabel('Emotions', fontsize=10)
  plt.ylabel('probability distribution ', fontsize=10)
  plt.xticks(index, emotion_ls, fontsize=10, rotation=30)
  plt.title('Distribution of emotions')
  plt.show()
  
  
# for i in wrong:
for i in correct:
  plot_img(i)

index=np.arange(7)
ls=[0,1,2,5,34,57]
# emotions=pd.file.iloc[]
y_pred=model.predict (x_test)


# In[3]:


import numpy as np
objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
y_pos = np.arange(len(objects))
print(y_pos)


# In[4]:


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
def emotion_analysis(emotions):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, emotions, align='center', alpha=0.9)
    plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
plt.show()


# In[11]:


# for static

from skimage import io
img = image.load_img('F:/Project Report/emotion detection/happy_ojas.jpeg', grayscale=True, target_size=(48, 48))
show_img=image.load_img('F:/Project Report/emotion detection/happy_ojas.jpeg', grayscale=False, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = loaded_model.predict(x)
#print(custom[0])
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(show_img)
plt.show()

m=0.000000000000000000001
a=custom[0]
for i in range(0,len(a)):
    if a[i]>m:
        m=a[i]
        ind=i
        
print('Expression Prediction:',objects[ind])
        


# In[5]:


mode="display"
import cv2


# In[ ]:


# emotions will be displayed on your face from the webcam feed
if mode == "display":
    loaded_model.load_weights('F:/Project Report/emotion detection/best_model_weights.hdf5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('F:\Project Report\emotion detection\Emotion-detection-master\Emotion-detection-master\src\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = loaded_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[26]:





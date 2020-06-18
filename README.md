# Facial-Expression-Recognition

In this project facial expression of the person was detected. Fer 2013 dataset was used, which consist of images of six basic emotional expressions including fear, disgust, anger, surprise, happiness,and sadness.

About the dataset
Fer2013 is an open-source dataset which is first, created for an ongoing project by Pierre-Luc Carrier and Aaron Courville, then shared publicly for a Kaggle competition, shortly before ICML 2013. This dataset consists of 35.887 grayscale, 48x48 sized face images with various emotions -7 emotions, all labeled-.
Emotion labels in the dataset:
0: -4593 images- Angry
1: -547 images- Disgust
2: -5121 images- Fear
3: -8989 images- Happy
4: -6077 images- Sad
5: -4002 images- Surprise
6: -6198 images- Neutral


Results

Confusion Matrix
                                 
                Angry  Disgust  Fear  Happy  Sad  Surprise Neutral  
         Angry  2111      29    331   330    932    148      595 
        Disgust 202      74     68     21    104     13      17 
         Fear   537       6     1145  323    1330   713      561 
        Happy   160       1     157   6707   507    172      341 
        Sad     484       5     302   436    3093    83      1092 
       Surprise 84        0     268   216    107    2826     108 
       Neutral  322       6     179   540    1101   113      3298 
       
       
             Precision  Recall F1-score 
       Angry  0.39      0.42    0.41 
     Disgust  0.95      0.99    0.97 
        Fear  0.45      0.38    0.38 
       Happy  0.68      0.69    0.69 
        Sad   0.44      0.38    0.41 
     Surprise 0.69      0.65    0.67 
      Neutral 0.45      0.49    0.47 
     Average  0.57      0.57    0.57 

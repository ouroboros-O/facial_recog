# Facial Recognition using Siamese Neural Networks


Decided to teach the sand how to think.

## Introduction: 

Solo Project by Muhammad Suhaib
*  **Dataset:** Labeled Faces in the Wild Home
. 
http://vis-www.cs.umass.edu/lfw/#download

## Structures and Frameworks 
 
*    **Python** 
*    **Tensorflow** for AI Machine Learning Libraries
*    **Kivy** for GUI
*    **Github** for collaboration
## Features

The application allows any user to first press a button to authenticate themselves in which the App with take a number of your pictures and then save them for the model to use as an anchor. Once this is done the user will then be able to authenticate and check whether or his face is being displayed in the webcam or not. At any point he can change and select another face or person to be the anchor by pressing that button.

## Contents

*   [Siamese Neural Networks](#Siamese-Neural-Networks)
*   [Download and Running](#Download-and-Running)
*   [Sources](#sources)
*   [Conclusion](#conclusion)

## Siamese Neural Networks:

A siamese neural network was used for this, in other words two identical data streams were set up. In one an incorrect image was compared with an anchor and in the other a unknown correct image was compared with an anchor image. These two models are then eventually brought together to come up with a single value once we train our algorithm.


## Download and Running:

To run the project, do the following:

Use git to clone this repository into your computer.

Download the trained model from this google drive link and place it into the following directories:
face\facial_recog\legacy
facial_recog\finalapp
https://drive.google.com/file/d/13ttCFEEoHbVT_MkWm0eeZnPf7qHFLd7k/view?usp=sharing



Run the py script in the finalapp folder.

If you wish to see the and or execute the scripts I used to compile the AI model you can see them in the legacy folder.



## Sources

[Kivy Tutorial by "Tech With Tim"] - "Used to Understand Kivy"

[Build a Python Facial Recognition App by Nicholas Renotte] - "Main source for understanding and execution"

[Siamese Neural Networks for One-shot Image Recognition] - https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf - "For the foundation of the techniques used"

## Conclusion

Overall it was an intense but extremely fun. Attempting to get the GPU to work with tensorflow so that the AI Model could be built in a timely fashion took an inordinate amount of time but overall I am very pleased with the results. We have good accuracy and a method that works. It's quite fascinating to see it working in real time.

## Pictures:
![63c82266-1591-4219-b1aa-4ebc1f675212](https://user-images.githubusercontent.com/90059140/235499079-17f91a94-9a3e-4e88-9dbb-9786ac19d6e9.jpg)

![6a721f2c-5406-481d-82f1-b019271d0853](https://user-images.githubusercontent.com/90059140/235499172-2e8fec21-c54b-4138-8746-f50d04f33bb9.jpg)

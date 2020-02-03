# algaeModel
The file algMod.py contains code for a convolutional neural network model and a transfer convolutional network model based off of the VGG16 model from the keras module. The model's purpose was to classify an image of a water surface as either containing algae or not. The classification of water surface images was part of a project initiative proposing to use the CNN to sort through images of a water reservoir taken by a drone in order to detect harmful algal blooms in a timely manner. 

Project Details

algMod.py was trained on a set of images showing water surfaces with and without algae contamination compiled both from online sources as well as a public database of a water reservoir. A small example of the images that were used for training, validation, and testing are in the folder algModPics which contains 100 images each of algae and non-algae photos. 

Prior to running algMod.py, make sure that the packages glob, os, scipy, tensorflow, sklearn, keras, and numpy are installed. 

To execute the file algMod.py, simply download the folder of images algModPics and update the path from which the program will retrieve the testing images from (located under the section "importing images for training, testing and validation"). The full set of images is not necessary given that the model has already been trained and saved as algMod.h5py with the full set of training images.

At your own discretion, you may also upload other images of water surfaces to be classified by algMod.py. In which case, the sections "Load realPredict image data -- no labels" and the function realTest could be used so long as the user changes the path name to the directory accordingly. 

Author

Julie Chen

Acknowledgements

https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python


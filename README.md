# IndoorSceneRecognition

### [Dataset](https://web.mit.edu/torralba/www/indoor.html)

There will be two models that will be built in this project:  
CNN - using keras and tensorflow  
Bayesian Logisitc Regression - R and manual  

## jupyter notebook files are run on google collab for more computation.

| File under models and training | Description |
| ----------- | ----------- |
| data_preprocessing.py | the python file that groups the MIT dataset into the 5 categories that we trained our models on |
| DS4420_CNN_feature_extractor.ipynb | the jupyter notebook that creates a CNN, trains it and then makes a feature extractor that sends results into two CSV files  |
| DS4420_Project_Bayesian.R | the R file that reads in the CSV files created by the CNN features extractor and then performs sampling to get the posterior, which is the likelihood of the labels given the features from the CNN|
| DS4420_Project_CNN.ipynb | two CNN models, the first one being identical to the CNN feature extractor, but the softmax layer is kept. The second model is another CNN that is trained specifically on the room types of the store category. It takes in images that are indoor scenes from stores and classify them into room types of stores|
| initial_POC_cnn_training.ipynb | the inital proof of concept file for phase 1 | 
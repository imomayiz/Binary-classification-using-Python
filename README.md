# ML-project
# Binary classification

## Brief description

In this project, we aim to apply different machine learning models onto two different datasets for binary classification. However, our scripts can be applied to different datasets.

## Algorithms

### Preprocessing

* Loads datasets
* Cleans data
* Splits datasets into training and test datasets 
* Additionnaly to cleaned data, outputs datasets on which we operated either tsne or PCA


### Classification 


We implement different models using Sklearn. The training is based on a GridSearch model that helps find the optimum parameters for a maximum 
accuracy across multiple validation sets using cross-validation.
It contains the following models:
* SVM
* KNN 
* Log regression
* Stochastic gradient descent


### Other classification models
* Neural networks
* Random forests
* Decision Trees

All these models are trained using the gridsearch method.

### Tests and results
In the program ```main.py```, we output the best accuracies reached by each model (fitted using gridsearch) and the corresponding 
optimum parameters. 


## Authors

* **Imane Momayiz** 
* **Guillaume Sallé**
* **Romain Namyst**
* **Thomas Kronland**
* **Karel Kedemos**







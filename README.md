Full Description

 The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively


Preprocess the Data

Using  knowledge of Pandas and scikit-learn’s StandardScaler(), we’ll need to preprocess the dataset. This step prepares you for Step 2, where we'll compile, train, and evaluate the neural network model.

We started this project by uploading file in Google Colab.
Process:
1. Read  in the charity_data.csv to a Pandas DataFrame,
2. Droped the EIN and NAME columns.
3. Determine the number of unique values in each column.
4. Using the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
5. Used pd.get_dummies() to encode categorical variables.

Compile, Train, and Evaluate the Model

Process:
1. Split the preprocessed data into a features array, X, and a target array, y. Used these arrays and the train_test_split function to split the data into training and testing datasets.
2. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.
3. Created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
4. Created the first hidden layer and choose an appropriate activation function.
5. Created an output layer with an appropriate activation function.
6. Compile and train the model.
7. Evaluated the model using the test data to determine the loss and accuracy.
8. Saved and export results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

Optimize the Model

Process:
Created a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.

. Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:

* Dropping more or fewer columns.
* Creating more bins for rare occurrences in columns.
* Increasing or decreasing the number of values for each bin.
* Add more neurons to a hidden layer.
* Add more hidden layers.
* Use different activation functions for the hidden layers.
* Add or reduce the number of epochs to the training regimen.
* Saved and export results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

- Write a Report on the Neural Network Model

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?
ans: IS_SUCCESSFUL variable which shows successful organization. This variable shows that the company's past funding was successful. 

What variable(s) are the features for your model?
ans: IS_SUCCESSFUL variable we have used for features for the model.


What variable(s) should be removed from the input data because they are neither targets nor features?
ans: We have removed 2 Input data that is EIN and NAME. Because they are not very useful.


Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
<img src="Images/neuron_layers.png">


Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?















# Master Project

This code was created to implement the Master´s Thesis Project investigating ADHD on social media using traditional machine learning, deep learning and transformer models by Hilde Mikaelsen and Vår Åsheim in the spring of 2024.

## Information About Repository

All data has been removed from the repository to follow GDPR rules. All filenames have been removed and needs to be replaced with the correct path to the file for running. Preprocessing files has to be run to create files to run models with. Some of the files included in this repository requires much memory and processing and has been run on NTNU High Performance Computer IDUN cluster to get results presented in the Master´s Thesis. The traditional and deep learning models use the embedddings created in preprocessing and the transformer models uses textual input also created in preprocessing.

Parts of the code including model setup, preprocsseing, text analysis, evaluation, plots and large file handling have been implemented assisted by ChatGPT 3.5/4.
This assistance includes bug fixing, suggesting frameworks and implementation stategies, as well as code suggestions. 


## ADHD Characteristics

This folder contains files to analyze the Reddit ADHD 2012 dataset. It consists of a preprocessing file and different files for analyzing various characteristics.

### Files Description

#### preprocessing_for_data.ipynb

This notebook reads a CSV file containing text data, preprocesses the text to remove stop words and punctuation, converts words to their base forms (lemmas), and saves the processed text back into a new CSV file. This is a necessary step to prepare the data for further analysis.

#### liwc.ipynb

Reads LIWC results, analyzes label distribution, calculates mean values of LIWC categories by label, identifies top LIWC categories for each label, and generates visual plots.

#### part_of_speech.ipynb

Tags POS using spaCy, cleans POS tags, analyzes POS tag distribution for each label, and plots the average number of each POS tag per post for each label.

#### sentiment.ipynb

Uses TextBlob to analyze sentiment, calculates average polarity and subjectivity, categorizes sentiments overall and by label, and visualizes sentiment distribution.

#### topic_modeling.ipynb

Apply topic modeling to discover themes. Uses TF-IDF vectorization and Latent Dirichlet Allocation (LDA) to extract topics from the dataset, and identifies topics for the entire dataset and for each label.

#### word_char_length.ipynb

Calculates average post lengths in words and characters for each label, generates distribution plots, and identifies maximum, mean, and cutoff values for word and character counts.

#### word_clouds.ipynb

Generates word clouds for the entire dataset and for each label to visualize the most frequently occurring words.

## Active Learning with modAL

This folder includes files that implements active learning strategies using the `modAL` library. It includes notebooks for data preprocessing and training various machine learning models.

### Files Description

#### preprocessing.ipynb

Reads the dataset from an Excel file (`ADHD2012.xlsx`).

- Removes duplicates, filters out invalid labels, and drops rows with missing or empty `title` and `selftext`. Removes usernames and URLs from the text data.
- Tokenizes the combined text (`title` + `selftext`) using NLTK.
- Maps text labels to integer values for modeling.
- Trains a Word2Vec model on the tokenized text and generates text embeddings.
- Splits the data into labeled and unlabeled datasets, and further into training and evaluation sets.
- Saves the preprocessed arrays and text data in pickle files for use with transformer models and traditional machine learning models.

#### Models folder

Each of the notebooks in this folder sets up and trains a specific machine learning model (Multi-Layer Perceptron (MLP) model, Naive Bayes, Random Forest, SVM) using the preprocessed data and an active learning loop.

- Loads preprocessed training, evaluation, and pool data from pickle files.
- Sets up the active learning loop using the `ActiveLearner` class with uncertainty sampling.
- Trains the MLP model, queries for new labels, updates the pool, and saves the active learner.
- Plots the accuracy over queries and saves predictions to a CSV file.

## Classification

This folder contains the classification models used for the Reddit ADHD Dataset created with active learning annotations. The folder contains three folders referring to traditional models, deep learning models and transformer models.

### Files Description

#### LogisticRegression.ipynb

This file implements Logistic Regression classifier using sklearn framework and evaluates using accuracy, recall, precision and F1. This file is used for classification of classes neutral, self-diagnosis and self-medication in the Reddit ADHD Dataset.

#### LogisticRegression_0and1.ipynb

This file implements Logistic Regression classifier using sklearn framework and evaluates using accuracy, recall, precision and F1. This file is used for classification of classes neutral and self-diagnosis in the Reddit ADHD Dataset.

#### multinomialNB.ipynb

This file implements Multinomial Naïve Bayes classifier using sklearn framework and evaluates using accuracy, recall, precision and F1. This file is used for classification of classes neutral, self-diagnosis and self-medication in the Reddit ADHD Dataset.

#### multinomialNB_0and1.ipynb

This file implements Multinomial Naïve Bayes classifier using sklearn framework and evaluates using accuracy, recall, precision and F1. This file is used for classification of classes neutral and self-diagnosis in the Reddit ADHD Dataset.

#### NaiveBayes.ipynb

This file implements Gaussian and Bernoulli Naïve Bayes classifier using sklearn framework and evaluates using accuracy, recall, precision and F1. This file is used for classification of classes neutral, self-diagnosis and self-medication in the Reddit ADHD Dataset.

#### NaiveBayes_0and1.ipynb

This file implements Gaussian and Bernoulli Naïve Bayes classifier using sklearn framework and evaluates using accuracy, recall, precision and F1. This file is used for classification of classes neutral and self-diagnosis in the Reddit ADHD Dataset.

#### RandomForest.ipynb

This file implements Random Forest classifier using sklearn framework and evaluates using accuracy, recall, precision and F1. This file is used for classification of classes neutral, self-diagnosis and self-medication in the Reddit ADHD Dataset.

#### RandomForest_0and1.ipynb

This file implements Random Forest classifier using sklearn framework and evaluates using accuracy, recall, precision and F1. This file is used for classification of classes neutral and self-diagnosis in the Reddit ADHD Dataset.

#### SVM.ipynb

This file implements linear and non-linear Support Vector Machine classifier using sklearn framework and evaluates using accuracy, recall, precision and F1. This file is used for classification of classes neutral, self-diagnosis and self-medication in the Reddit ADHD Dataset.

#### SVM_0and1.ipynb

This file implements linear and non-linear Support Vector Machine classifier using sklearn framework and evaluates using accuracy, recall, precision and F1. This file is used for classification of classes neutral and self-diagnosis in the Reddit ADHD Dataset.

#### ann.ipynb

Implementation of Feed-forward artificial neural network using tensorflow for classification of classes neutral, self-diagnosis and self-medication in Reddit ADHD Dataset created with active learning annotations.

#### ann_01.ipynb

Implementation of Feed-forward artificial neural network using tensorflow for classification of classes neutral and self-diagnosis in Reddit ADHD Dataset created with active learning annotations.

#### cnn.py

Implementation of Convolutional Neural Network using tensorflow framework. This file was run on high performance computer as it requires much memory and processing power.

#### Albert.py

Implementation of ALBERT transformer pre-trained "albert-base-v2" model from https://huggingface.co/albert/albert-base-v2, using PyTorch framework. This file was run using high performance computer as it requires much memory and processing power. This file was used for classication task with classes neutral, self-diagnosis and self-medication, as well as for binary classification of neutral and self-diagnosis by changing the file paths and number of labels in instantiation of model.

#### Bert.py

Implementation of BERT transformer pre-trained "bert-base-uncased" model from https://huggingface.co/google-bert/bert-base-uncased, using PyTorch framework. This file was run using high performance computer as it requires much memory and processing power. This file was used for classication task with classes neutral, self-diagnosis and self-medication, as well as for binary classification of neutral and self-diagnosis by changing the file paths and number of labels in instantiation of model.

#### distilBert.py

Implementation of DistilBERT transformer pre-trained "distilbert-base-uncased" model from https://huggingface.co/distilbert/distilbert-base-uncased, using PyTorch framework. This file was run using high performance computer as it requires much memory and processing power. This file was used for classication task with classes neutral, self-diagnosis and self-medication, as well as for binary classification of neutral and self-diagnosis by changing the file paths and number of labels in instantiation of model.

#### Roberta.py

Implementation of RoBERTa transformer pre-trained "roberta-base" model from https://huggingface.co/FacebookAI/roberta-base, using PyTorch framework. This file was run using high performance computer as it requires much memory and processing power. This file was used for classication task with classes neutral, self-diagnosis and self-medication, as well as for binary classification of neutral and self-diagnosis by changing the file paths and number of labels in instantiation of model.

#### XLNet.py

Implementation of XLNet transformer pre-trained "xlnet-base-cased" model from https://huggingface.co/xlnet/xlnet-base-cased, using PyTorch framework. This file was run using high performance computer as it requires much memory and processing power. This file was used for classication task with classes neutral, self-diagnosis and self-medication, as well as for binary classification of neutral and self-diagnosis by changing the file paths and number of labels in instantiation of model.

## SMHD

This folder includes preprocessing of the SMHD dataset with classes neutral and ADHD. Classification models trained on this dataset are implemented in this folder.

### Folder classificationSMHD

This folder contains folders deeplearning, traditional and transformers which includes classification models for SMHD dataset classification task of classes Neutral and ADHD.

#### ANN-SMHD.ipynb

Implementation of Feed-forward artificial neural network using tensorflow.

#### LogisticRegressionSMHD.ipynb

This file implements Logistic Regression classifier using sklearn framework and evaluates using accuracy, recall, precision and F1.

#### NaiveBayesSMHD.ipynb

This file implements Gaussian, Bernoulli and Multinomial Naïve Bayes classifiers using sklearn framework and evaluates using accuracy, recall, precision and F1.

#### RF-SMHD.ipynb

This file implements Random Forest classifier using sklearn framework and evaluates using accuracy, recall, precision and F1.

#### SVM-SMHD.ipynb

This file implements linear and non-linear Support Vector Machine classifiers using sklearn framework and evaluates using accuracy, recall, precision and F1.

#### Albert_smhd.py

Implementation of ALBERT transformer pre-trained "albert-base-v2" model from https://huggingface.co/albert/albert-base-v2, using PyTorch framework. This file was run using high performance computer as it requires much memory and processing power.

#### Bert_smhd.py

Implementation of BERT transformer pre-trained "bert-base-uncased" model from https://huggingface.co/google-bert/bert-base-uncased, using PyTorch framework. This file was run using high performance computer as it requires much memory and processing power.

#### distilBERT_smhd.py

Implementation of DistilBERT transformer pre-trained "distilbert-base-uncased" model from https://huggingface.co/distilbert/distilbert-base-uncased, using PyTorch framework. This file was run using high performance computer as it requires much memory and processing power.

#### Roberta_smhd.py

Implementation of RoBERTa transformer pre-trained "roberta-base" model from https://huggingface.co/FacebookAI/roberta-base, using PyTorch framework. This file was run using high performance computer as it requires much memory and processing power.

#### XLnet_smhd.py

Implementation of XLNet transformer pre-trained "xlnet-base-cased" model from https://huggingface.co/xlnet/xlnet-base-cased, using PyTorch framework. This file was run using high performance computer as it requires much memory and processing power.

### Files Description

#### limitedSMHD.py

Extract a limited subset of the SMHD dataset. Limits the dataset to 100 entries per condition. Writes it to a JSON file to use in experiments.

#### preprocessing.ipynb

Preprocess the SMHD dataset for analysis and modeling.

- Loads the limited dataset from a JSON file (`train_smhd_limited.json`).
- Splits the 'posts' column into separate rows for detailed analysis.
- Separates the data into ADHD and control groups.
- Tokenizes the text data using NLTK.
- Generates text embeddings using Word2Vec.
- Splits the dataset into training and test sets, ensuring balanced representation.
- Saves the processed data into pickle files for use in various machine learning models.

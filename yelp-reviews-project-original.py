# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [markdown] Cell 1
# # YELP R PROJECT
#
# YELP R:
#
# * Website: https://huggingface.co/datasets/yelp_review_full
# * Paper: https://arxiv.org/abs/1509.01626
# * Description: Very large collection of product reviews with star ratings.
# * Task: Star rating prediction, sentiment analysis

# %% [markdown] Cell 2
# ## Useful Imports

# %% [markdown] Cell 3
# We prepare the notebook environment importing the necessary libraries to run the code.

# %% [code] Cell 4
# def install(package):
#     import sys
#     import subprocess
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package, "--quiet"])

# # %% [code] Cell 5
# install("accelerate")
# install("contractions")
# install("datasets")
# install("gensin")
# install("nltk")
# install("torch")
# install("transformers")

# %% [markdown] Cell 6
# The **imported** packages are:
# * **re**: used for regular expressions
# * **warnings**: used to ignore annoying and meaningless warnings
# * **contractions**: used for text preprocessing, e.g., "That's" becomes "That is"
# * **numpy**: used for mathematical and algebrical operations
# * **pandas**: used for DataFrame management
# * **string**: used to import the punctuation, that will define the regex for text preprocessing
# * **datasets**: used to load the "yelp-review-full" dataset
# * **collections**: used to count words occurrences in the corpus
# * **gensim**: used to download the GloVe model through an API call
# * **nltk**: used for stopwords setting and simple tokenizer for text preprocessing
# * **tensorflow**: used to build the neural network model

# %% [code] Cell 7
import os
import random
import string
import subprocess
import sys
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict, load_dataset
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Fix randomness and hide warnings
seed = int(os.environ.get("PYTHONHASHSEED", 42))

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %% [markdown] Cell 8
# ## Preliminary Analysis:
#

# %% [markdown] Cell 9
# ### Analyze the Dataset
# Briefly describe the dataset:
# * what type of documents does it contain
# * how many documents are there
# * calculate and visualise some simple statistics for the collection, e.g., the average document length, the average vocabulary size, etc.
#

# %% [markdown] Cell 10
# Load the dataset from Hugging Face. The Yelp reviews dataset consists of reviews from Yelp. It is extracted from the Yelp Dataset Challenge 2015 data.

# %% [code] Cell 11
dataset = load_dataset("yelp_review_full", split="train")
testset = load_dataset("yelp_review_full", split="test")

# %% [markdown] Cell 12
#  A typical data point, comprises of a text and the corresponding label:
# - 'text': The review texts are Yelp reviews about everything, from doctors, to restaurants, bars, beauty salons and other services.
# - 'label': Corresponds to the score associated with the review (between 0 and 4 stars).
#
# The Yelp reviews full dataset is constructed by randomly taking 130,000 training samples and 10,000 testing samples for each review star from 0 to 4, having a balanced dataset. In total there are 650,000 trainig samples and 50,000 testing samples.

# %% [code] Cell 13
dataset_df = pd.DataFrame(dataset)

dataset_df

# %% [code] Cell 14
testset_df = pd.DataFrame(testset)

testset_df

# %% [markdown] Cell 15
# In the following part we analyse the dataset computing some statistics both on the training set and the test set, to check if the document length and the vocabulary size are on average the same in the two collections.
#
# We notice that the average document length is around 730 characters in both collections.
# We also notice that we have almost the same average vocabulary per document in the test set as in the dataset used for training, but we have a smaller vocabulary in the test set considering the total words in all the documents, which is expected, having a much smaller set of data.

# %% [code] Cell 16
# Average document length in the dataset
avg_doc = 0

for i in range(0, dataset.shape[0]):
    avg_doc += len(dataset[i]["text"])

avg_doc = avg_doc / dataset.shape[0]

avg_doc

# %% [code] Cell 17
# Average document length in the testset
avg_doc = 0

for i in range(0, testset.shape[0]):
    avg_doc += len(testset[i]["text"])

avg_doc = avg_doc / testset.shape[0]

avg_doc

# %% [code] Cell 18
# %%time  # (magic command commented out)
# Average vocabulary size in the dataset
regex = "[" + string.punctuation + "]"

total_vocab = set()
avg_vocab = 0

tmp_df = pd.Series()
tmp_df = dataset_df["text"].str.replace(pat=regex, repl="", regex=True)

for i in range(0, dataset_df.shape[0]):
    doc = tmp_df.iloc[i]
    words = set(doc.lower().split())
    total_vocab.update(words)
    avg_vocab += len(words)

avg_vocab = avg_vocab / dataset.shape[0]
total_vocab = total_vocab
tot_vocab = len(total_vocab)

print(avg_vocab)
print(tot_vocab)

del tmp_df

# %% [markdown] Cell 19
# In principle we could also remove the stopwords from the data before computing the statistics, but they wouldn't influence our results.
#
# In our dataset the reviews are mainly in english, so the stopwords that will be removed are the ones of the english vocabulary.
#
# In any case, given that in the stopwords are contained also words such as: "and", "but", "not", and given that these words could enforce a concept or change it in a completely different way when reading a review, we will not remove the stopwords when dealing with complex training models (the ones using DistilBERT).

# %% [code] Cell 20
# %%time  # (magic command commented out)
nltk.download("stopwords")

s_words = stopwords.words("english")
total_vocab = [w for w in total_vocab if w not in s_words]
tot_vocab = len(set(total_vocab))

tot_vocab

# %% [code] Cell 21
# %%time  # (magic command commented out)
# Average vocabulary size in the testset
regex = "[" + string.punctuation + "]"

total_vocab = set()
avg_vocab = 0

tmp_df = pd.Series()
tmp_df = testset_df["text"].str.replace(pat=regex, repl="", regex=True)

for i in range(0, testset.shape[0]):
    doc = tmp_df.iloc[i]
    words = set(doc.lower().split())
    total_vocab.update(words)
    avg_vocab += len(words)

avg_vocab = avg_vocab / testset.shape[0]
tot_vocab = len(set(total_vocab))

print(avg_vocab)
print(tot_vocab)

del tmp_df

# %% [markdown] Cell 22
# # Training models

# %% [markdown] Cell 23
# In this part of the notebook, we will show the models that we decided to use to perform multi-class classification and predict the associated rating for new data points. We will evaluate different models and compare their performance.
#
# We start from a simple model, a linear classifier with Bag of Words, that we will then use as baseline in comparison with more complex models, such as DistilBERT.

# %% [markdown] Cell 24
# We first recreate the dataframe, such that, if in other parts of the notebook we modified the data we can start from a clean situation and execute this training model. We will perform this operation before any new paragraph also to facilitate if someone wants to execute only a part of the code.

# %% [code] Cell 25
dataset_df = pd.DataFrame(dataset)

dataset_df

# %% [code] Cell 26
testset_df = pd.DataFrame(testset)

testset_df

# %% [markdown] Cell 27
# ## Baseline - Simple Classifiers

# %% [markdown] Cell 28
# ### Logistic Regression with BoW feature vector

# %% [markdown] Cell 29
# We need to convert the text data into feature values that can be given to the classifier.
# Let's use the most common feature, which is the frequency of the words, because the vocabulary present in a document provides strong signal about the meaning of the document and so the category it belongs to.
#
# Using Bag of Words (BoW) we represent the documents as vectors of word counts.
#
# We use `CountVectorizer` to extract this representation, fitting it on the training data, so it will decide which is the vocabulary of the collection. Then we will apply it on the data to generate the bag of words.

# %% [code] Cell 30
train_x = [txt for txt in dataset_df["text"]]
label_x = [lbl for lbl in dataset_df["label"]]

# %% [code] Cell 31
vectorizer = CountVectorizer()
vectorizer.fit(train_x)

# %% [code] Cell 32
# The number of features extracted by CountVectorizer fitted on the training data
print("Vocabulary size: ", len(vectorizer.get_feature_names_out()))

# %% [code] Cell 33
# Let's look at some of the features - we notice there are a lot of not useful features,
# unusual and rare words so we will limit the vocabulary
vectorizer.get_feature_names_out()[:50]

# %% [code] Cell 34
# We recreate the CountVectorizer, limiting the vocabulary, removing stopwords and
# taking only words that appear in more than 50 documents
vectorizer = CountVectorizer(min_df=50, stop_words="english", lowercase=True)
vectorizer.fit(train_x)

# We can notice that in this way the vocabulary is reduced a lot
print("Vocabulary size: ", len(vectorizer.get_feature_names_out()))

# %% [code] Cell 35
# The features still seem unusual and not meaningful, but given the min_df = 50,
# these are not rare words for our dataset
vectorizer.get_feature_names_out()[:100]

# %% [code] Cell 36
# Vector representation of each document, using BoW, storing data in a Sparse Matrix to do not run out of memory
train_x_vector = vectorizer.transform(train_x)

train_x_vector

# %% [markdown] Cell 37
# Once we have prepared the data as a BoW to have the features to give in input to the model, we can build and train a simple classifier, that we will use as a baseline for our project, trying then to achieve better results with more complex models.

# %% [code] Cell 38
model = LogisticRegression().fit(train_x_vector, label_x)

# %% [markdown] Cell 39
# We can visualize the elements with highest positive coefficients for the different classes, influencing the most the predictions, and we can notice that many of these words are adjectives and in particular, for class '0', very strong and negative adjectives. In fact, when the ratings are low and the client is disappointed, he tends to use a high number of adjectives to emphasize the dissatisfaction, expressing sentiments and opinions also with an emotional tone. On the other side for the class '4' we can notice that there are some positive adjectives that influence a lot the predictions, but as we can see some really positive adjectives, such as 'perfection' are also present in class '3', and even in higher position, and this can be one of the reasons the model doesn't reach high performance, giving similar weights to the same features for different classes.
#
# Another problem in our dataset, that we noticed by analyzing some reviews, is that even if the content is mainly negative or mainly positive, it is really subjective how the rating star is associated, and this is one of the causes of not such high performance of our models.

# %% [code] Cell 40
vocab = vectorizer.get_feature_names_out()

for i, label in enumerate(set(label_x)):
    top10 = np.argsort(model.coef_[i])[-10:][::-1]
    if i == 0:
        top = pd.DataFrame(vocab[top10], columns=[label])
        top_indices = top10
    else:
        top[label] = vocab[top10]
        top_indices = np.concatenate((top_indices, top10), axis=None)

print(top)

# %% [markdown] Cell 41
# Let's now perform the prediction on the training set and inspect the model.

# %% [code] Cell 42
test_x = [txt for txt in testset_df["text"]]
test_y = [lbl for lbl in testset_df["label"]]

# %% [code] Cell 43
test_vector = vectorizer.transform(test_x)
predictions = model.predict(test_vector)

# %% [markdown] Cell 44
# Let's evaluate the model: `LogisticRegression` for multiclass classification with bag of words.

# %% [code] Cell 45
print("Model accuracy: ", accuracy_score(predictions, test_y))
print("\nClassification report:\n")
print(
    classification_report(test_y, predictions, target_names=["0", "1", "2", "3", "4"])
)

# %% [code] Cell 46
cm = confusion_matrix(test_y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# %% [markdown] Cell 47
# From the performance metrics of the model and from the confusion matrix we can conclude that:
#
# - The accuracy of the model is 58%, which is better than random guessing, which would yield 20% accuracy for a 5 class problem, but there is still room for improvement.
# - The model performs best for class '0', the negative reviews, but the model also performs relatively well for class '4', functioning best at the extremes of the possible value spectrum, than in the middle, indicating more difficulty in distinguishing these middle ratings.
# - From the confusion matrix we can also see that the model frequently confuses negative reviews ('1') with very negative reviews ('0') and at the other side of the spectrum it tends to misclassify positive reviews ('3') with very positive reviews ('4'), tending always to the extremes of the spectrum of values. So, we can say that the model has an underestimation in lower ratings and an overestimation in higher ratings, implying a bias of the model towards more negative predictions in the lower part of the ratings spectrum and a bias towards more positive predictions in the upper part of the ratings values.
#
# To improve these areas of the model, we can enhance the features extraction techniques, using TF-IDF, word embeddings with more complex models and advanced techniques like DistilBERT.

# %% [markdown] Cell 48
# Given the obtained results, we will keep as baseline the Logistic Regression, without regularization, with BoW used for the input features.
# The obtained accuracy is of 58%.

# %% [code] Cell 49
trainset = load_dataset("yelp_review_full", split="train")
testset = load_dataset("yelp_review_full", split="test")

train_texts = trainset["text"]
train_labels = trainset["label"]

print(len(train_texts), len(train_labels))

# %% [code] Cell 50
target_classes = [0, 1, 2, 3, 4]
test_texts = testset["text"]
test_labels = testset["label"]

# %% [markdown] Cell 51
# ### Logistic Regession with TF-IDF feature vector

# %% [code] Cell 52
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.5, min_df=50, lowercase=True, stop_words="english"
)

X_train = tfidf_vectorizer.fit_transform(train_texts)

# %% [markdown] Cell 53
# Let's inspect the length of the vocabulary

# %% [code] Cell 54
feature_names = tfidf_vectorizer.get_feature_names_out()

print(len(feature_names))

# %% [code] Cell 55
model = LogisticRegression().fit(X_train, train_labels)

# %% [markdown] Cell 56
# It can be observed that the use of TF-IDF in place of BoW results in the identification of different words, although some of the previously identified overlapping problems remain.

# %% [code] Cell 57
vocab = tfidf_vectorizer.get_feature_names_out()

for i, label in enumerate(set(train_labels)):
    top10 = np.argsort(model.coef_[i])[-10:][::-1]
    if i == 0:
        top = pd.DataFrame(vocab[top10], columns=[label])
        top_indices = top10
    else:
        top[label] = vocab[top10]
        top_indices = np.concatenate((top_indices, top10), axis=None)

print(top)

# %% [code] Cell 58
X_test = tfidf_vectorizer.transform(test_texts)
predictions = model.predict(X_test)

# %% [code] Cell 59
target_names = ["0", "1", "2", "3", "4"]

print("Model accuracy: ", accuracy_score(predictions, test_labels))
print("\nClassification report:\n")
print(classification_report(test_labels, predictions, target_names=target_names))

# %% [code] Cell 60
# Compute the confusion matrix
cm = confusion_matrix(test_labels, predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm.T,
    xticklabels=target_classes,
    yticklabels=target_classes,
    cmap="Blues",
    annot=True,
)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")
plt.show()

# %% [markdown] Cell 61
# The transition from BoW to TF-IDF in our Logistic Regression model resulted in a marginal improvement in accuracy. While traditional methods such as TF-IDF offer certain advantages, we will attempt to achieve further enhancements by leveraging advanced representations and more complex models, including word embeddings (GloVe) and contextual embeddings derived from transformer-based models (DistilBERT), which could potentially lead to more pronounced performance improvements.
#
# We will start from these considerations to better improve the performance on the text classification task for Yelp reviews in the next sections of the notebook.

# %% [markdown] Cell 62
# ## Character-level CNN
#
# In this section, we implement the Character-level Convolutional Neural Network model presented in the paper.

# %% [code] Cell 63
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from Datasets import load_dataset
from keras.layers import (
    Activation,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    MaxPooling1D,
)
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras

# %% [code] Cell 64
# Retrieve the training set

trainset = load_dataset("yelp_review_full", split="train")

# %% [markdown] Cell 65
# ### Preprocessing: convert text to sequence of indices
#
# In this section, we preprocess the text from the reviews. Following the approach outlined in the referenced paper, we perform character-level tokenization on the texts after converting all the characters to lowercase. Using a tokenizer from keras.preprocessing, we map each character in the text to its corresponding index in the tokenizer's vocabulary. The alphabet used for tokenization is the one mentioned in the paper, consisting of 70 characters:  ``abcdefghijklmnopqrstuvwxyz0123456789\u2014,;.!?:'"/\\|_@#$%^&*~\`+-=<>()[]{}\n``

# %% [code] Cell 66
trainset_df = pd.DataFrame(trainset)

train_texts = trainset_df["text"].str.lower().values
train_labels = trainset_df["label"].values

target_classes = list(trainset_df["label"].unique())
target_classes.sort()

tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK")
tk.fit_on_texts(train_texts)

# %% [code] Cell 67
# construct a new vocabulary
alphabet = (
    "abcdefghijklmnopqrstuvwxyz0123456789\u2014,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
)
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1

print("Size of the considered alphabet: ", len(alphabet))

# %% [code] Cell 68
print("Original tokenizer indexing:\n", tk.word_index)

# Create new character indexing according to the created vocabulary
# So, we use char_dict to replace the tk.word_index
tk.word_index = char_dict.copy()

# also update the inverse-map
tk.index_word = {value: key for key, value in char_dict.items()}

print("\nNew indexing:\n", tk.word_index)

# %% [markdown] Cell 69
# We add the UNK token (representing unknown characters) to the vocabulary, resulting in a vocabulary size of 71 (70 characters + UNK). We map the UNK token to the index 0 because, in the embedding layer, we will be reserving the index 0 for padding and will be mapped to the null vector. This ensures that both padding and unknown characters do not affect the learning process, as they are represented by the same neutral vector.

# %% [code] Cell 70
# Add 'UNK' to the vocabulary obtaining a vocabulary of size 70 + UNK
tk.word_index[tk.oov_token] = 0
tk.index_word[0] = tk.oov_token

print("Final indexing:\n", tk.word_index)

# %% [code] Cell 71
# convert reviews into sequences of tokens
train_sequences = tk.texts_to_sequences(train_texts)

print(train_texts[0])
print("\nTokenized text:")
print(train_sequences[0])

del train_texts

# %% [markdown] Cell 72
# We perform padding to ensure that all sequences have a uniform length of 1014. The paper suggests performing backward quantization, where the latest characters are placed near the beginning of the output, making it easier for fully connected layers to associate weights with the latest readings. By using `padding='pre'`, the padding is added to the beginning of the sequences, ensuring that the input does not start with all zero values during backward quantization.

# %% [code] Cell 73
train_data = pad_sequences(
    train_sequences, maxlen=1014, padding="pre", value=tk.word_index[tk.oov_token]
)

# Convert to numpy array to facilitate computation
train_data = np.array(train_data, dtype="int64")
train_data = train_data[:, ::-1]

print("Training dataset shape: ", train_data.shape)
del train_sequences

# %% [code] Cell 74
# Let's look at how the sequences will be given to the model
print(train_data[:5])

# %% [markdown] Cell 75
# We perform one-hot encoding of the labels.

# %% [code] Cell 76
train_classes = train_labels
# train_class_list = [x - 1 for x in train_classes]
# Use classes from 0 to 4, which represent the rating
train_class_list = [x for x in train_classes]

# One-hot encoding of the labels
from keras.utils import to_categorical

train_classes = to_categorical(train_class_list)

print(train_classes[:5])

# %% [code] Cell 77
del trainset_df

# %% [markdown] Cell 78
# ### Helper functions

# %% [code] Cell 79
# A set of functions to reuse some of the fucntionalities explained in the notebook


def init_tokenizer(alphabet):

    tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK")

    tk.fit_on_texts(alphabet)

    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1

    inverted_char_dict = {value: key for key, value in char_dict.items()}

    tk.word_index = char_dict.copy()
    tk.index_word = inverted_char_dict

    # Add 'UNK' to the vocabulary obtaining a vocabulary of size 70
    tk.word_index[tk.oov_token] = 0
    tk.index_word[0] = tk.oov_token

    print("Vocabulary length: %d" % len(tk.word_index))
    print("Final indexing:\n", tk.word_index)

    return tk


def preprocess_feature_vectors(texts, tokenizer):
    train_sequences = tokenizer.texts_to_sequences(texts)

    # Padding - to ensure all sequences have same lenght - 1014
    # With 'padding = pre', the padding is added to the beginning of the sequences
    # Chosen because of the backward quantization, to place the latest characters near the beginning of the output
    train_data = pad_sequences(
        train_sequences,
        maxlen=1014,
        padding="pre",
        value=tokenizer.word_index[tokenizer.oov_token],
    )

    # Convert to numpy array to facilitate computation
    train_data = np.array(train_data, dtype="int64")
    train_data = train_data[:, ::-1]

    return train_data


def preprocess_labels(labels):
    train_classes = labels
    # train_class_list = [x - 1 for x in train_classes]
    # Use classes from 0 to 4, which represent the rating
    train_class_list = [x for x in train_classes]

    # One-hot encoding of the labels

    train_classes = to_categorical(train_class_list)

    return train_classes


# %% [markdown] Cell 80
# ### Character-level CNN architecture
#
# Below we define the hyperparameter and architecture for the Small char-level-CNN.
#
# The model takes an input of size 1014, which is likely the length of the input sequences. It consists of the following layers:
#
# 1. **Embedding Layer**: The input is passed through an embedding layer, which maps the input indices to their one-hot encoding.
#
# 2. **Convolutional Layers**: The model has 6 convolutional layers, each with 256 filters and varying filter sizes (7, 7, 3, 3, 3, 3). Some of the convolutional layers are followed by max-pooling layers with a pool size of 3
#
# 3. **Fully Connected Layers**: After the convolutional layers, the features are flattened and passed through two fully connected layers, each with 1024 units, with a dropout rate of 0.5 applied.
#
# 4. **Output Layer**: The final layer is a dense layer with 5 units (corresponding to the number of classes) and a softmax activation function.
#
# The model is compiled using the Adam optimizer and categorical cross-entropy loss function. The embedding layer is initialized with one-hot encoding weights.
#
# We chose Keras as the framework to build the model because our team had prior experience with it in building Convolutional Neural Networks.
#
# We also tried the _large_ variant of the network and the results were consistent with the ones presented in the papers, namely the increase in performance was negligible, so we opted to show only the execution and results of the small variant.

# %% [code] Cell 81
# parameter
input_size = 1014

conv_layers = [
    [256, 7, 3],
    [256, 7, 3],
    [256, 3, -1],
    [256, 3, -1],
    [256, 3, -1],
    [256, 3, 3],
]

fully_connected_layers = [1024, 1024]
num_of_classes = 5
dropout_p = 0.5

# The paper uses SGD but Adam has been proven to perform
# better in nearly all scenarios
optimizer = "adam"
loss = "categorical_crossentropy"

# %% [markdown] Cell 82
# In the cell below, we use the `Embedding` class from `Keras` to build a one-hot encoder that can be integrated as a Keras component when constructing the model. This approach eliminates the need to one-hot encode the entire dataset and store it in dense form. Since Keras primarily operates on dense tensors and has limited support for sparse tensors, pre-encoding the dataset would result in an excessively large dataset and significant memory waste, thus we opted to perform the encoding during the training phase.

# %% [code] Cell 83
vocab_size = len(tk.word_index) - 1  # -1 to not consider the UNK token
embedding_size = vocab_size

# Embedding weights
embedding_weights = np.identity(vocab_size + 1)

# removing the first column so the first row is all zeros so that tokens
# mapped to 0, namely padding and the UNK token, are associated with a 0
# vector
embedding_weights = embedding_weights[:, 1:]

embedding_weights = np.array(embedding_weights)
print("Load")
print("Shape of embedding_weights:", embedding_weights.shape)
print("Expected size: (", vocab_size + 1, embedding_size, ")")

# Embedding layer Initialization
# We set the trainable variable to False, in this way the weights will not change
# during training and the layer will embed the input to a one-hot encoded vector
# for each saple at each step in the training.
onehot_embedding_layer = Embedding(vocab_size + 1, embedding_size, trainable=False)


# %% [code] Cell 84
# Model Construction
# Input
def get_compiled_model(embedding_layer):
    inputs = Input(shape=(input_size,), name="input")  # shape=(?, 1014)

    # Embedding
    x = embedding_layer(inputs)

    # Conv
    for filter_num, filter_size, pooling_size in conv_layers:
        x = Conv1D(filter_num, filter_size)(x)
        x = Activation("relu")(x)
        if pooling_size != -1:
            x = MaxPooling1D(pool_size=pooling_size)(x)  # Final shape=(None, 34, 256)

    x = Flatten()(x)  # (None, 8704)
    # Fully connected layers
    for dense_size in fully_connected_layers:
        x = Dense(dense_size, activation="relu")(x)  # dense_size == 1024
        x = Dropout(dropout_p)(x)

    # Output Layer
    predictions = Dense(num_of_classes, activation="softmax")(x)

    # Build model
    model = Model(inputs=inputs, outputs=predictions)

    # intializinig the weights of the embedding layer to obtain the
    # onehot encoding
    onehot_embedding_layer.set_weights([embedding_weights])

    model.compile(
        optimizer=optimizer, loss=loss, metrics=["accuracy"]
    )  # Adam, categorical_crossentropy

    return model


# %% [code] Cell 85
model = get_compiled_model(onehot_embedding_layer)
model.summary()

# %% [markdown] Cell 86
# Here we verify the correctness of the encoding with the Embedding module from Keras. This layer can be used as an input-output function only after the model has been compiled.

# %% [code] Cell 87
# checking the correctness of the encoding

# getting the encoding layer
enc = model.layers[1]
print("Layer: ", enc)

# a is the first token in the vocabulary - mapped to [1 0 ... 0]
# \n is the last token in the vocabulary - mapped to [0 ... 0 1]
# £ is out of vocabulary - mapped to [0 ... 0]
test_string = "aa\n£"
seq = np.array(tk.texts_to_sequences([test_string])[0])
print("Input feature vector: ", seq)

emb = enc(seq).numpy()
print("\nOne-Hot encoding of the input vector:")
print(emb)

# %% [markdown] Cell 88
# ### Training

# %% [code] Cell 89
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)

x_train = train_data[indices]
y_train = train_classes[indices]

print(x_train.shape, y_train.shape)

# Define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=4, mode="max", restore_best_weights=True
)


# Training
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    batch_size=128,
    epochs=10,
    verbose=1,
    callbacks=[early_stopping],
).history

# %% [code] Cell 90
model.save("Small-char-level-cnn.keras")
del x_train, y_train, train_data, train_classes

# %% [code] Cell 91
# Find the epoch with the highest validation accuracy
best_epoch = np.argmax(history["val_accuracy"])

# Plot training and validation performance metrics
plt.figure(figsize=(20, 5))

# Plot training and validation loss
plt.plot(history["loss"], label="Training", alpha=0.8, color="#ff7f0e", linewidth=3)
plt.plot(
    history["val_loss"], label="Validation", alpha=0.8, color="#4D61E2", linewidth=3
)
plt.legend(loc="upper left")
plt.title("Categorical Crossentropy")
plt.grid(alpha=0.3)

plt.figure(figsize=(20, 5))

# Plot training and validation accuracy, highlighting the best epoch
plt.plot(history["accuracy"], label="Training", alpha=0.8, color="#ff7f0e", linewidth=3)
plt.plot(
    history["val_accuracy"], label="Validation", alpha=0.8, color="#4D61E2", linewidth=3
)
plt.plot(
    best_epoch,
    history["val_accuracy"][best_epoch],
    marker="*",
    alpha=0.8,
    markersize=10,
    color="#4D61E2",
)
plt.legend(loc="upper left")
plt.title("Accuracy")
plt.grid(alpha=0.3)

plt.show()

del history

# %% [markdown] Cell 92
# ### Model Evaluation

# %% [code] Cell 93
testset = load_dataset("yelp_review_full", split="test")

# %% [code] Cell 94
testset_df = pd.DataFrame(testset)

test_texts = testset_df["text"].str.lower().values
test_labels = testset_df["label"].values

# convert reviews into sequences of tokens
test_sequences = preprocess_feature_vectors(test_texts, tk)

# Predict labels for the entire test set
predictions = np.argmax(model.predict(test_sequences, verbose=0), axis=-1)

# Display the shape of the predictions
print("Predictions Shape:", predictions.shape)

# %% [code] Cell 95
print("Model accuracy: ", accuracy_score(predictions, test_labels))
print("\nClassification report:\n")
print(
    classification_report(
        test_labels, predictions, target_names=["0", "1", "2", "3", "4"]
    )
)

# %% [code] Cell 96
# Compute the confusion matrix
cm = confusion_matrix(test_labels, predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm.T,
    xticklabels=target_classes,
    yticklabels=target_classes,
    cmap="Blues",
    annot=True,
)
plt.ylabel("True labels")
plt.xlabel("Predicted labels")
plt.show()

del testset_df, model

# %% [markdown] Cell 97
# The CNN achieves a modest improvement in performance across all evaluation metrics compared to Logistic Regression, with an average increase of 5%. This increase demonstrates that by looking only at a the sequence of characters, the CNN can grasp the intrinsic semantic meaning of the text being examined, enabling it to perform sentiment analysis with discreate accuracy.
#
# While the CNN model achieves marginally better performance, this improvement comes at the cost of significantly longer training times. The added complexity of the CNN architecture, with its convolutional and fully connected layers, requires more computational resources and a longer training process. This tradeoff between the marginal performance gain and the increased training overhead begs the question of whether the CNN model is worth using instead of the simpler Logistic Regression approach.

# %% [markdown] Cell 98
# ### Trying a different alphabet
#
# In the following section we repeat the above experiment by training the network on a different alphabet which discriminates between lower and uppercase letters.

# %% [code] Cell 99
# We need to re-instantiate the trainset because in the first experiment we lowercased the text
trainset_df = trainset.to_pandas()

train_texts = trainset_df["text"].values
train_labels = trainset_df["label"].values

# we can see that now the text has uppercase characters
print(train_texts[1])

del trainset_df

# %% [code] Cell 100
alphabet_upper = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789—,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
tk_upper = init_tokenizer(alphabet_upper)

# %% [code] Cell 102
train_data = preprocess_feature_vectors(train_texts, tk_upper)
train_classes = preprocess_labels(train_labels)

del train_texts, train_labels

# %% [code] Cell 103
vocab_size = len(tk_upper.word_index) - 1  # -1 to not consider the UNK token
embedding_size = vocab_size

# Embedding weights
embedding_weights = np.identity(vocab_size + 1)

# removing the first column so the first row is all zeros so that tokens
# mapped to 0, namely padding and the UNK token, are associated with a 0
# vector
embedding_weights = embedding_weights[:, 1:]

embedding_weights = np.array(embedding_weights)
print("Load")
print("Shape of embedding_weights:", embedding_weights.shape)
print("Expected size: (", vocab_size + 1, embedding_size, ")")

# Embedding layer Initialization
# We set the trainable variable to False, in this way the weights will not change
# during training and the layer will embed the input to a one-hot encoded vector
# for each saple at each step in the training.
onehot_embedding_layer = Embedding(vocab_size + 1, embedding_size, trainable=False)

# %% [code] Cell 104
model = get_compiled_model(onehot_embedding_layer)
model.summary()

# %% [code] Cell 105
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)

x_train = train_data[indices]
y_train = train_classes[indices]

print(x_train.shape, y_train.shape)

# Define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=4, mode="max", restore_best_weights=True
)


# Training
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    batch_size=128,
    epochs=10,
    verbose=1,
    callbacks=[early_stopping],
).history

# %% [code] Cell 106
model.save("Small-char-level-cnn-uppercase-alphabet.keras")
del x_train, y_train, train_data, train_classes

# %% [code] Cell 107
# Find the epoch with the highest validation accuracy
best_epoch = np.argmax(history["val_accuracy"])

# Plot training and validation performance metrics
plt.figure(figsize=(20, 5))

# Plot training and validation loss
plt.plot(history["loss"], label="Training", alpha=0.8, color="#ff7f0e", linewidth=3)
plt.plot(
    history["val_loss"], label="Validation", alpha=0.8, color="#4D61E2", linewidth=3
)
plt.legend(loc="upper left")
plt.title("Categorical crossentropy")
plt.grid(alpha=0.3)

plt.figure(figsize=(20, 5))

# Plot training and validation accuracy, highlighting the best epoch
plt.plot(history["accuracy"], label="Training", alpha=0.8, color="#ff7f0e", linewidth=3)
plt.plot(
    history["val_accuracy"], label="Validation", alpha=0.8, color="#4D61E2", linewidth=3
)
plt.plot(
    best_epoch,
    history["val_accuracy"][best_epoch],
    marker="*",
    alpha=0.8,
    markersize=10,
    color="#4D61E2",
)
plt.legend(loc="upper left")
plt.title("Accuracy")
plt.grid(alpha=0.3)

plt.show()

del history

# %% [markdown] Cell 108
# #### Model Evaluation

# %% [code] Cell 109
testset_df = testset.to_pandas()

test_texts = testset_df["text"].values
test_labels = testset_df["label"].values

# convert reviews into sequences of tokens
test_sequences = preprocess_feature_vectors(test_texts, tk)

# Predict labels for the entire test set
predictions = np.argmax(model.predict(test_sequences, verbose=0), axis=-1)

# Display the shape of the predictions
print("Predictions Shape:", predictions.shape)

# %% [code] Cell 110
print("Model accuracy: ", accuracy_score(predictions, test_labels))
print("\nClassification report:\n")
print(
    classification_report(
        test_labels, predictions, target_names=["0", "1", "2", "3", "4"]
    )
)

# %% [code] Cell 111
# Compute the confusion matrix
cm = confusion_matrix(test_labels, predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm.T,
    xticklabels=target_classes,
    yticklabels=target_classes,
    cmap="Blues",
    annot=True,
)
plt.ylabel("True labels")
plt.xlabel("Predicted labels")
plt.show()

# %% [markdown] Cell 112
# As can be seen, increasing the alphabet was detrimental for the task, resulting in overall worse performance compared to the model trained with the smaller alphabet.

# %% [markdown] Cell 113
# ## Custom LSTM and Glove

# %% [markdown] Cell 114
# In this paragraph we comment the preprocessing and a simple model built for the review classification task.

# %% [markdown] Cell 115
# ### Initialization: Install and Imports

# %% [markdown] Cell 116
# First of all, we **installed** the packages used throughout the code.

# %% [code] Cell 117
# !pip3 install datasets contractions nltk gensim  # (magic command commented out)

# %% [markdown] Cell 118
# The **imported** packages are:
# * **re**: used for regular expressions
# * **warnings**: used to ignore annoying and meaningless warnings
# * **contractions**: used for text preprocessing, e.g., "That's" becomes "That is"
# * **numpy**: used for mathematical and algebrical operations
# * **pandas**: used for DataFrame management
# * **string**: used to import the punctuation, that will define the regex for text preprocessing
# * **datasets**: used to load the "yelp-review-full" dataset
# * **collections**: used to count words occurrences in the corpus
# * **gensim**: used to download the GloVe model through an API call
# * **nltk**: used for stopwords setting and simple tokenizer for text preprocessing
# * **tenforflow**: used to build the neural network model

# %% [code] Cell 119
import re
import string
import warnings
from collections import Counter
from string import punctuation

import contractions
import numpy as np
import pandas as pd
from datasets import load_dataset

# import gensim.downloader as api

# ignore annoying warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

# %% [code] Cell 120
import nltk

nltk.download("stopwords")
nltk.download("punkt")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stopwords = set(stopwords.words("english"))

# %% [code] Cell 121
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

# %% [markdown] Cell 122
# The first layer of the neural network is represented by an Embedding layer, which we build using the GloVe model.
#
# Our neural network will leverage these embeddings to convert words into dense vector representations that capture semantic meaning. To do that, we load the pre-trained GloVe model using the gensim library's API.

# %% [code] Cell 123
embed_vector_len = 300

glove_model = api.load(f"glove-wiki-gigaword-{embed_vector_len}")

# %% [markdown] Cell 124
# ### Dataset Download and Text Preprocessing

# %% [markdown] Cell 125
# In this section we define how we'll preprocess the texts of the reviews, we'll load the dataset and we'll put it in the right format to be given in input to the neural network.
#
# The functions used for the preprocessing phase are two:
# * ***remove_noise***: formed by the following lines of code:
#     * lowercasing: Converts all characters in the text to lowercase, ensuring uniformity
#     * removing punctuation: eliminates punctuation marks from the text, as they do not contribute to the meaning in our classification task
#     * handling contractions: expands contractions to their full forms (e.g., "That's" becomes "That is"), which can help in standardizing the text and improving the accuracy of tokenization
#     * tokenization: splits the text into individual words or tokens
#     * removing stopwords: filters out common stopwords (e.g., "and", "the", "is"), which are words that typically do not carry significant meaning and can be safely removed to reduce noise
#     * reconstructing the text: joins the filtered tokens back into a single string, forming the preprocessed text
# * ***sequence_padding***: preprocesses our text data by converting it into sequences of tokens and ensuring uniform length through padding and truncation
#     * the *tokenizer.texts_to_sequences(sentences)* method converts each sentence in the list *sentences* into a sequence of integers. Each integer corresponds to the index of a word in the tokenizer's vocabulary
#     * the *sequence.pad_sequences* method adjusts the length of each sequence to *max_sentence_length*
#     * if a sequence is shorter than *max_sentence_length*, it will be padded with zeros at the end of the sequence (*padding=post*)
#     * if a sequence is longer than max_sentence_length, it will be truncated to fit this length from the end of the sequence (*truncating=post*)
#
# Then we have model's creation. We comment its architecture:
# * input layer: *Input(shape=(max_sentence_length,))* defines the input shape, which corresponds to the padded sequence length of each sentence
# * embedding layer: it is built using the embedding of the most relevant words, as will be shown later. It is initialized using the embedding matrix formed by the single embedding vectors returned by the GloVe model. This operation is done by the *embedding_layer.set_weights([emb_matrix])*. The embedding layer is set to non-trainable to leverage pre-trained GloVe weights without updating them during training
# * bidirectional LSTM layer: allowing the model to capture dependencies in both forward and backward directions in the text
# * dense layers: three fully connected layers with ReLU activation functions are added to the neural netowrk to learn complex representations and increase network's depth
# * output layer: the output size is specified by the parameter in input. We show two models: one with 5 classes in output (one for each star for review rating), and one with 3 classes (positive, neutral, and negative review)

# %% [code] Cell 126
# length of the vector representing the review
max_sentence_length = 300
# if the review is shorter, we add zeros in the end
padding = "post"
# if the review is longer, we truncate the words over the 300th
truncating = "post"

# activation function used for the output layer of the neural network
activation = "softmax"
# loss function used by the neural network
loss = "categorical_crossentropy"


# function used to remove the punctuation from the text, substituting it with a white space
def remove_punctuation(text):
    punct_regex = "[" + string.punctuation + "]"
    return re.sub(punct_regex, " ", text)


# preprocessing function of a text
def remove_noise(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = remove_punctuation(text)
    # remove contractions (e.g., That's become That is)
    text = contractions.fix(text)
    # list of words in the text
    tokens = nltk.word_tokenize(text)
    # remove stopwords
    tokens = [word for word in tokens if word not in stopwords]
    # build the text back
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


# function used to convert the text to a sequence of defined length, applies padding and truncating
def sequence_padding(sentences, tokenizer):
    sequences = tokenizer.texts_to_sequences(sentences)
    pad_trunc_sequences = sequence.pad_sequences(
        sequences, maxlen=max_sentence_length, padding=padding, truncating=truncating
    )

    return pad_trunc_sequences


# function used to map the labels from the instances in the dataset for a 3 class classification
def map_label(label):
    # 5 and 4 stars reviews are mapped to value 2
    if label > 2:
        return 2
    # 3 stars reviews are mapped to value 1
    elif label == 2:
        return 1
    # 2 and 1 stars reviews are mapped to value 0
    else:
        return 0


# function used to create the neural network's model
def create_model(vocab_size, emb_matrix, output_size):
    model = Sequential()

    # define input shape
    model.add(Input(shape=(max_sentence_length,)))

    # build embedding layer with defined parameters
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embed_vector_len,
        input_length=max_sentence_length,
        # not trainable
        trainable=False,
    )
    embedding_layer.build((1,))
    # set the weights of the embedding layer to the values returned by the GloVe model
    embedding_layer.set_weights([emb_matrix])
    model.add(embedding_layer)

    model.add(Bidirectional(LSTM(256)))

    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))

    model.add(Dense(output_size, activation=activation))

    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

    return model


# %% [markdown] Cell 127
# We download the trainset and the testset of the yelp_review_full dataset.

# %% [code] Cell 128
trainset = load_dataset("yelp_review_full", split="train")
testset = load_dataset("yelp_review_full", split="test")

# %% [markdown] Cell 129
# We convert them to DataFrames.

# %% [code] Cell 130
trainset_df = pd.DataFrame(trainset)
testset_df = pd.DataFrame(testset)

# %% [markdown] Cell 131
# We remove noise from the reviews by applying the *remove_noise* preprocessing function, defined above.

# %% [code] Cell 132
trainset_df["text"] = trainset_df["text"].apply(remove_noise)
testset_df["text"] = testset_df["text"].apply(remove_noise)

# %% [markdown] Cell 133
# In this section, we initialize a tokenizer and fit it on the text data from the training set. The tokenizer will convert words to numerical indices. Notice that we set the out of vocabulary token of the tokenizer to "UNK".

# %% [code] Cell 134
features = trainset_df["text"]

tokenizer = Tokenizer(oov_token="UNK")
tokenizer.fit_on_texts(features)
word_index = tokenizer.word_index

# %% [markdown] Cell 135
# We count the occurrences of all the words in the corpus, and we keep only the ones which count is higher then the defined threshold. Then we build the new dictionary of the word_index attribute of the tokenizer where we save only the most relevant words, assigning them new indices starting from 1. The index 0 is then assigned to the out of vocabulary token.

# %% [code] Cell 136
word_frequencies = Counter(word for text in features for word in text.split())
min_word_frequency = 50
common_words = [
    word for word, freq in word_frequencies.items() if freq >= min_word_frequency
]

word_index_filtered = {word: index for index, word in enumerate(common_words, start=1)}
vocab_size_filtered = len(word_index_filtered) + 1

tokenizer.word_index = word_index_filtered
tokenizer.word_index[tokenizer.oov_token] = 0

# %% [markdown] Cell 137
# In this section, we create an embedding matrix that maps each word in our tokenizer's vocabulary to its corresponding GloVe embedding vector. This matrix will be used to initialize the embedding layer in our neural network.
#
# We comment the main concepts of the piece of code below:
# * *emb_matrix.append(np.zeros(embed_vector_len))* adds an array of zeros representing the embedding vector for the unknown token 'UNK'. This ensures that the 'UNK' token has a placeholder embedding
# * for each element in the dictionary of the tokenizer, we check whether the word exists in the GloVe model, and we append the retrieved embedding vector to the embedding matrix
# * the elif branch is taken in cases in which the word being analyzed is not recognized by the GloVe model but it is not the "UNK" token. In this case we manage it by adding an array of zeros, meaning the word does not carry any meaningful information
#
# Finally, we put the built embedding matrix to the correct format to be given as inizialization matrix for the Embedding layer's weights.

# %% [code] Cell 138
emb_matrix = []

# this array of zeros represents the embedding vector for the unknown character
emb_matrix.append(np.zeros(embed_vector_len))

# for each couple in the items of the (new) word_index of the tokenizer
for word, index in tokenizer.word_index.items():
    # if the word is recognized by the GloVe model
    if word in glove_model:
        # we get its embedding vector and append it to the embedding matrix
        embedding_vector = glove_model.get_vector(word)
        emb_matrix.append(embedding_vector)
    # if it is not unkown, we insert an array of zeros
    elif word != "UNK":
        emb_matrix.append(np.zeros(embed_vector_len))

# %% [code] Cell 139
emb = np.array(emb_matrix)
emb.shape

# %% [markdown] Cell 140
# ### Review Classification on Stars

# %% [markdown] Cell 141
# In this section we apply what we have seen above: we preprocess the data, we build the model, we train, and evaluate it.
#
# In particular, this section is used for the classification based on the number of stars of the review.

# %% [code] Cell 142
train_x = trainset_df["text"]
train_y = trainset_df["label"]
test_x = testset_df["text"]
test_y = testset_df["label"]

# %% [code] Cell 143
labels = [0, 1, 2, 3, 4]

# %% [markdown] Cell 144
# We apply the *sequence_padding* function on reviews' text, putting them in the correct format.

# %% [code] Cell 145
train_x = sequence_padding(train_x, tokenizer)
test_x = sequence_padding(test_x, tokenizer)

# %% [markdown] Cell 146
# Since we use *categorical_crossentropy* as loss function, we need to convert the targets to a one-hot encoding representation. To do that, we use the *to_categorical* function.

# %% [code] Cell 147
train_oh_y = to_categorical(train_y, num_classes=len(labels))
test_oh_y = to_categorical(test_y, num_classes=len(labels))

# %% [markdown] Cell 148
# We create the model using the function defined above.

# %% [code] Cell 149
model_5 = create_model(vocab_size_filtered, emb, 5)
model_5.summary()

# %% [code] Cell 150
callbacks_5 = [
    EarlyStopping(monitor="val_loss", patience=1),
    ModelCheckpoint("model_5.keras", save_best_only=True, save_weights_only=False),
    ReduceLROnPlateau(patience=1),
]

# %% [markdown] Cell 151
# We train the neural network model using the training data.

# %% [code] Cell 152
history_5 = model_5.fit(
    train_x,
    train_oh_y,
    epochs=10,
    batch_size=32,
    verbose=1,
    validation_split=0.2,
    callbacks=callbacks_5,
)

# %% [markdown] Cell 153
# We evaluate the performance of the trained model on the testset.

# %% [code] Cell 154
train_loss, train_accuracy = model_5.evaluate(train_x, train_oh_y)

print("Final Performance on the Training Set:")
print("\tAccuracy:", train_accuracy)
print("\tLoss:", train_loss)
print("\n")

test_loss, test_accuracy = model_5.evaluate(test_x, test_oh_y)

print("Performance on the Test Set:")
print("\tAccuracy:", test_accuracy)
print("\tLoss:", test_loss)

# %% [markdown] Cell 155
# ### Conclusions

# %% [markdown] Cell 156
# As shown by the confusion matrix below, the diagonal dominates. The problem is with the cells adjacent to the diagonal, which show a lot of misprediction between adjacent classes, by both underrating and overrating. In particular we notice that the model struggles to predict 4 stars rating, often assigning to the review 5 stars, as we can see from the cell on the last line, showing 3000 mispredictions. This is probably due to the text features, which are partly shared between classes 4 and 5.

# %% [code] Cell 157
predictions = model_5.predict(test_x)

# %% [code] Cell 158
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(test_y, predictions.argmax(axis=1))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm.T, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=True)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")
plt.show()

del model_5

# %% [markdown] Cell 159
# ## DistilBERT: Fine Tuning
# [BERT](https://huggingface.co/docs/transformers/model_doc/bert) is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.
# However, the [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) model has been used because it has 60% the dimension of BERT, while retaining 97% of its language understanding capabilities and being 60% faster.
# The pre-trained DistilBERT model is fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, without substantial task-specific architecture modifications.


# %% [code] Cell 160
def install(package):
    """
    Install a Python package.
    """
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", package, "--quiet"]
    )


# %% [markdown] Cell 161
# In this section, we've used the [PyTorch](https://pytorch.org/) framework for loading DistilBERT model and built upon it a Fine-Tuned Version of DistilBERT.
# In particular, we used [PyTorch-Ignite](https://pytorch.org/ignite/index.html) package that provides engine, handlers, and metrics for training and evaluating neural networks in PyTorch.

# %% [code] Cell 162
install("numpy")
install("datasets")
install("torch")
install("transformers")
install("torchtext")
install("pytorch-ignite")

# %% [markdown] Cell 163
# ### Import All You Need

# %% [code] Cell 164
from os import environ, getcwd
from random import seed
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, ConfusionMatrix, Precision, Recall
from ignite.utils import manual_seed
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer

# %% [markdown] Cell 165
# We defined `DISCARD` to eventually discard the provided percentage of the training dataset: it was useful in the preliminary steps when we wanted to speed up the training step.
# Then, we defined the ratio between the size of the training and validation sets with `RATIO_TRAIN_EVAL`.

# %% [code] Cell 166
DISCARD = 0.0
RATIO_TRAIN_EVAL = 0.75

TRAIN = (1 - DISCARD) * RATIO_TRAIN_EVAL
EVAL = 1 - DISCARD - TRAIN

BATCH_SIZE = 8
LR = 5e-5
NUM_CLASSES = 5
NUM_EPOCHS = 1
PATIENCE = 2
SEED = 42

environ["PYTHONHASHSEED"] = str(SEED)
environ["MPLCONFIGDIR"] = getcwd() + "/configs/"

filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=Warning)

np.random.seed(SEED)
seed(SEED)

# %% [code] Cell 167
manual_seed(SEED)

# %% [markdown] Cell 168
# ### Basic Setup

# %% [markdown] Cell 169
# #### Data Preprocessing
#
# The YelpReviewFull dataset is loaded by mean of the [`load_dataset`](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/loading_methods#datasets.load_dataset) function of the [`datasets`](https://huggingface.co/docs/datasets/index) Python-package.
#
# Then we load the DistilBERT's tokenizer because we noticed how the usage of the model specific's tokenizer is suggested by almost all the text classification models.

# %% [code] Cell 170
raw_datasets = load_dataset("yelp_review_full")

# %% [code] Cell 171
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# %% [code] Cell 172
def tokenize_function(examples):
    """
    Tokenize the text field of the dataset.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Mapping is for high performance in the tokenization process
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# %% [markdown] Cell 173
# Once the dataset has been tokenized, the `'text'` field became useless to the model, so we discarded it, and then we renamed the field `'label'` into `'labels'`.
#
# Once we've processed the absolute sizes for the training and validation sets, we split the training set into training and validation sets.
# The testing set is entirely loaded as it is, we just shuffled it.

# %% [code] Cell 174
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_size = int(TRAIN * len(tokenized_datasets["train"]))
test_size = int(EVAL * len(tokenized_datasets["train"]))
discard_size = len(tokenized_datasets["train"]) - train_size - test_size

small_train_dataset, small_eval_dataset, _ = random_split(
    tokenized_datasets["train"].shuffle(seed=SEED),
    [train_size, test_size, discard_size],
)
small_test_dataset = tokenized_datasets["test"].shuffle(seed=SEED)

# %% [code] Cell 175
print(
    "Train size: %s\nEval size: %s\nTest size: %s"
    % (len(small_train_dataset), len(small_eval_dataset), len(small_test_dataset))
)

# %% [markdown] Cell 176
# #### Dataloaders
#
# As mentioned, we used the PyTorch-Ignite package for training DistilBERT, but it requires to use more efficient data objects.
#
# [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) combines a dataset and a sampler, and provides an iterable over the given dataset: we convert the training, validation and testing sets into dataloaders divided in batches.

# %% [code] Cell 177
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(small_test_dataset, batch_size=BATCH_SIZE)

# %% [markdown] Cell 178
# ### Model
#
# [`AutoModelForSequenceClassification`](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification) is a generic model class we use to instantiate the classification model with name `'distilbert-base-uncased'` from the [`transformer`](https://huggingface.co/transformers/v3.0.2/index.html) Python-library.

# %% [code] Cell 179
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=NUM_CLASSES
)

# %% [markdown] Cell 180
# #### Optimizer
#
# The model is compiled using:
# * [`AdamW`](https://huggingface.co/transformers/v3.0.2/main_classes/optimizer_schedules.html?highlight=adamw#transformers.AdamW) as optimizer with learning rate equal to `5e-5`,
# * [`PieceWiseLinear`](https://pytorch.org/ignite/v0.4.7/generated/ignite.handlers.param_scheduler.PiecewiseLinear.html) learning rate scheduler.

# %% [code] Cell 181
optimizer = AdamW(model.parameters(), lr=LR)

# %% [code] Cell 182
num_training_steps = NUM_EPOCHS * len(train_dataloader)

milestones_values = [(0, 5e-5), (num_training_steps, 0.0)]

lr_scheduler = PiecewiseLinear(
    optimizer, param_name="lr", milestones_values=milestones_values
)

# %% [markdown] Cell 183
# #### Set Device

# %% [code] Cell 184
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# %% [markdown] Cell 185
# ### Create Trainer
#
# The trainer is instance of the class [`Engine`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine), which runs the function `train_step` over each batch of a dataset and emits events as it goes.
# In particular, we related the event `ITERATION_STARTED` with the learning rate scheduler [`add_event_handler`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.add_event_handler), even if it was not determinant because the number of epochs is fixed to one.


# %% [code] Cell 186
def train_step(engine, batch):
    """
    Train step function.
    """
    model.train()
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss


# %% [code] Cell 187
trainer = Engine(train_step)

# %% [code] Cell 188
trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

# %% [code] Cell 189
pbar = ProgressBar()

# %% [code] Cell 190
pbar.attach(trainer, output_transform=lambda x: {"loss": x})

# %% [markdown] Cell 191
# ### Create Evaluator
#
# Once we created the engine for the training step, we also implemented the engines for the evaluation steps (over the training set and over the validation set after each epoch has ended), and we attached the [`Accuracy`](https://pytorch.org/ignite/generated/ignite.metrics.Accuracy.html) metric to them.


# %% [code] Cell 192
def evaluate_step(engine, batch):
    """
    Evaluate step function.
    """
    model.eval()

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits

    return {"y_pred": logits, "y": batch["labels"]}


# %% [code] Cell 193
train_evaluator = Engine(evaluate_step)
validation_evaluator = Engine(evaluate_step)

# %% [code] Cell 194
Accuracy().attach(train_evaluator, "accuracy")
Accuracy().attach(validation_evaluator, "accuracy")


# %% [code] Cell 195
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    """
    Log training results.
    """
    train_evaluator.run(train_dataloader)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    print(
        f"Training Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}"
    )


def log_validation_results(engine):
    """
    Log validation results.
    """
    validation_evaluator.run(eval_dataloader)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    print(
        f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}"
    )


trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

# %% [markdown] Cell 196
# After the definition of the metric we will use, we add the [`EarlyStopping`](https://pytorch.org/ignite/generated/ignite.handlers.early_stopping.EarlyStopping.html) object to the handlers of the validation evaluator.
#
# Among the handlers available in PyTorch-Ignite, we also attached the [`ModelCheckpoint`](https://pytorch.org/ignite/generated/ignite.handlers.checkpoint.ModelCheckpoint.html) handler to periodically save objects to the disk.


# %% [code] Cell 197
def score_function(engine):
    """
    Score function for the EarlyStopping handler.
    """
    val_accuracy = engine.state.metrics["accuracy"]
    return val_accuracy


handler = EarlyStopping(
    patience=PATIENCE, score_function=score_function, trainer=trainer
)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)

# %% [code] Cell 198
checkpointer = ModelCheckpoint(
    dirname="model", filename_prefix="distilbert-yelp", n_saved=2, create_dir=True
)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {"models": model})

# %% [markdown] Cell 199
# ### Model Training
#
# We decided to run the training phase for just one epoch because
# 1. each epoch is time-demanding when it ran over the entire dataset ($\approx 650.000$ samples),
# 2. the model's improvement after consecutive epochs is null, e.g., the accuracy does not increase with two epochs of training, and
# 3. we outperformed the models in the paper "[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)" at first try.
#
# For these reasons, we assumed that DistilBERT would not remarkably improve after the first epoch, hence, it should be our time-horizon.

# %% [code] Cell 200
trainer.run(train_dataloader, max_epochs=NUM_EPOCHS)

# %% [markdown] Cell 201
# #### Performance Assessment
#
# Finally, we evaluate the model over the testing set.

# %% [code] Cell 202
tester = Engine(evaluate_step)
pbar_test = ProgressBar()
pbar_test.attach(tester, output_transform=lambda x: {"loss": x})
Accuracy().attach(tester, "accuracy")
ConfusionMatrix(num_classes=NUM_CLASSES).attach(tester, "cm")
Recall(average=True).attach(tester, "recall")
Precision(average=True).attach(tester, "precision")
tester.run(test_dataloader)
metrics = tester.state.metrics
avg_accuracy = metrics["accuracy"]

print(
    f"Testing Results - Avg accuracy: %.3f\nTesting Results - Avg recall: %f\nTesting Results - Avg precision: %f"
    % (avg_accuracy, metrics["recall"], metrics["precision"])
)

# %% [code] Cell 203
ConfusionMatrixDisplay(
    confusion_matrix=metrics["cm"].numpy(), display_labels=range(1, NUM_CLASSES + 1)
).plot()
plt.show()

# %% [markdown] Cell 204
# ### Conclusions
#
# Some considerations about the results.
# * The fine-tuned version of DistilBERT has outperformed the models described in the "[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)" paper by achieving an accuracy of **0.684**.
# * The model performs best for class '0', the negative reviews, but the model also performs relatively well for class '4', functioning best at the extremes of the possible value spectrum, than in the middle, indicating more difficulty in distinguishing these middle ratings.
# * The model has an underestimation in lower ratings and an overestimation in higher ratings, implying a bias of the model towards more negative predictions in the lower part of the ratings spectrum and a bias towards more positive predictions in the upper part of the ratings values.
#
# Even if the results are the highest over the YelpReviewFull dataset, the performances highlight how the models are not able to learn from the dataset.
# This might be because the dataset is too noisy - we have to remember the dataset is composed of reviews written by people, which are sometimes too subjective and not always coherent with the rating they assigned - or because the classification task is not correct for the dataset.
# One possible try could be to change the classification task into a regression task, where the model has to predict the rating as a continuous value, instead of a discrete one: in this way, the model could learn better the dataset because it might discover, for example, reviews of rating 0 are much more similar to reviews of rating 1 than to reviews of rating 4.

# %% [markdown] Cell 205
# ## DistilBERT: Transfer Learning
#
# In this section, we repeat the same exact steps we did for training DistilBERT in the previous section, but now we freeze the layers of the DistilBERT model, and we train only the classification head: this is the typical transfer learning approach.


# %% [code] Cell 206
def install(package):
    """
    Install the required package
    """
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", package, "--quiet"]
    )


# %% [markdown] Cell 207
# In this section, we've used the [PyTorch](https://pytorch.org/) framework for loading DistilBERT model and building upon it a transfer learning version of DistilBERT.
# In particular, we used [PyTorch-Ignite](https://pytorch.org/ignite/index.html) package that provides engine, handlers, and metrics for training and evaluating neural networks in PyTorch.

# %% [code] Cell 208
install("numpy")
install("datasets")
install("torch")
install("transformers")
install("torchtext")
install("pytorch-ignite")

# %% [markdown] Cell 209
# ### Import All You Need

# %% [code] Cell 210
from os import environ, getcwd
from random import seed
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, ConfusionMatrix, Precision, Recall
from ignite.utils import manual_seed
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer

# %% [markdown] Cell 211
# We defined `DISCARD` to eventually discard the provided percentage of the training dataset: it was useful in the preliminary steps when we wanted to speed up the training step.
# Then, we defined the ratio between the size of the training and validation sets with `RATIO_TRAIN_EVAL`.

# %% [code] Cell 212
DISCARD = 0.0
RATIO_TRAIN_EVAL = 0.75

TRAIN = (1 - DISCARD) * RATIO_TRAIN_EVAL
EVAL = 1 - DISCARD - TRAIN

BATCH_SIZE = 8
LR = 5e-5
NUM_CLASSES = 5
NUM_EPOCHS = 1
PATIENCE = 2
SEED = 42

environ["PYTHONHASHSEED"] = str(SEED)
environ["MPLCONFIGDIR"] = getcwd() + "/configs/"

filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=Warning)

np.random.seed(SEED)
seed(SEED)

# %% [code] Cell 213
manual_seed(SEED)

# %% [markdown] Cell 214
# ### Basic Setup

# %% [markdown] Cell 215
# #### Data Preprocessing
#
# The YelpReviewFull dataset is loaded by mean of the [``load_dataset``](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/loading_methods#datasets.load_dataset) function of the [``datasets``](https://huggingface.co/docs/datasets/index) Python-package.
#
# Then we load the DistilBERT's tokenizer because we noticed how the usage of the model specific's tokenizer is suggested by almost all the text classification models.

# %% [code] Cell 216
raw_datasets = load_dataset("yelp_review_full")

# %% [code] Cell 217
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# %% [code] Cell 218
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Mapping is for high performance in the tokenization process
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# %% [markdown] Cell 219
# Once the dataset has been tokenized, the ``'text'`` field became useless to the model, so we discarded it, and then we renamed the field ``'label'`` into ``'labels'``.
#
# Once we've processed the absolute sizes for the training and validation sets, we split the training set into training and validation sets.
# The testing set is entirely loaded as it is, we just shuffled it.

# %% [code] Cell 220
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_size = int(TRAIN * len(tokenized_datasets["train"]))
test_size = int(EVAL * len(tokenized_datasets["train"]))
discard_size = len(tokenized_datasets["train"]) - train_size - test_size

small_train_dataset, small_eval_dataset, _ = random_split(
    tokenized_datasets["train"].shuffle(seed=SEED),
    [train_size, test_size, discard_size],
)
small_test_dataset = tokenized_datasets["test"].shuffle(seed=SEED)

# %% [code] Cell 221
print(
    f"Train size: {len(small_train_dataset)}\nEval size: {len(small_eval_dataset)}\nTest size: {len(small_test_dataset)}"
)

# %% [markdown] Cell 222
# #### Dataloaders
#
# As mentioned in the header of the notebook, we used the PyTorch-Ignite package for training DistilBERT, but it requires to use more efficient data objects.
#
# [``DataLoader``](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) combines a dataset and a sampler, and provides an iterable over the given dataset: we convert the training, validation and testing sets into dataloaders divided in batches.

# %% [code] Cell 223
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(small_test_dataset, batch_size=BATCH_SIZE)

# %% [markdown] Cell 224
# ### Model
#
# [``AutoModelForSequenceClassification``](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification) is a generic model class we use to instantiate the classification model with name ``'distilbert-base-uncased'`` from the [``transformer``](https://huggingface.co/transformers/v3.0.2/index.html) Python-library.

# %% [code] Cell 225
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=NUM_CLASSES
)

# %% [code] Cell 226
for param in model.distilbert.parameters():
    param.requires_grad = False

for i, param in enumerate(model.parameters()):
    print(f"{i}.\t{param.size()}\t-\tFreezed: {param.requires_grad}")

# %% [markdown] Cell 227
# #### Optimizer
#
# The model is compiled using:
# * [``AdamW``](https://huggingface.co/transformers/v3.0.2/main_classes/optimizer_schedules.html?highlight=adamw#transformers.AdamW) as optimizer with learning rate equal to ``5e-5``.
# * [``PieceWiseLinear``](https://pytorch.org/ignite/v0.4.7/generated/ignite.handlers.param_scheduler.PiecewiseLinear.html) learning rate scheduler

# %% [code] Cell 228
optimizer = AdamW(model.parameters(), lr=LR)

# %% [code] Cell 229
num_training_steps = NUM_EPOCHS * len(train_dataloader)

milestones_values = [(0, 5e-5), (num_training_steps, 0.0)]

lr_scheduler = PiecewiseLinear(
    optimizer, param_name="lr", milestones_values=milestones_values
)

# %% [markdown] Cell 230
# #### Set Device

# %% [code] Cell 231
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# %% [markdown] Cell 232
# ### Create Trainer
#
# The trainer is instance of the class [``Engine``](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine), which runs the function ``train_step`` over each batch of a dataset and emits events as it goes.
# In particular, we related the event ``ITERATION_STARTED`` with the learning rate scheduler [``add_event_handler``](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.add_event_handler), even if it was not determinant because the number of epochs is fixed to one.


# %% [code] Cell 233
def train_step(engine, batch):
    model.train()
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss


# %% [code] Cell 234
trainer = Engine(train_step)

# %% [code] Cell 235
trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

# %% [code] Cell 236
pbar = ProgressBar()

# %% [code] Cell 237
pbar.attach(trainer, output_transform=lambda x: {"loss": x})

# %% [markdown] Cell 238
# ### Create Evaluator
#
# Once we created the engine for the training step, we also implemented the engines for the evaluation steps (over the training set and over the validation set after each epoch has ended), and we attached the [``Accuracy``](https://pytorch.org/ignite/generated/ignite.metrics.Accuracy.html) metric to them.


# %% [code] Cell 239
def evaluate_step(engine, batch):
    model.eval()

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits

    return {"y_pred": logits, "y": batch["labels"]}


# %% [code] Cell 240
train_evaluator = Engine(evaluate_step)
validation_evaluator = Engine(evaluate_step)

# %% [code] Cell 241
Accuracy().attach(train_evaluator, "accuracy")
Accuracy().attach(validation_evaluator, "accuracy")


# %% [code] Cell 242
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_dataloader)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    print(
        f"Training Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}"
    )


def log_validation_results(engine):
    validation_evaluator.run(eval_dataloader)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    print(
        f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}"
    )


trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

# %% [markdown] Cell 243
# After the definition of the metric we will use, we add the [``EarlyStopping``](https://pytorch.org/ignite/generated/ignite.handlers.early_stopping.EarlyStopping.html) object to the handlers of the validation evaluator.
#
# Among the handlers available in PyTorch-Ignite, we also attached the [``ModelCheckpoint``](https://pytorch.org/ignite/generated/ignite.handlers.checkpoint.ModelCheckpoint.html) handler to periodically save objects to the disk.


# %% [code] Cell 244
def score_function(engine):
    val_accuracy = engine.state.metrics["accuracy"]
    return val_accuracy


handler = EarlyStopping(
    patience=PATIENCE, score_function=score_function, trainer=trainer
)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)

# %% [code] Cell 245
checkpointer = ModelCheckpoint(
    dirname="model",
    filename_prefix="distilbert-yelp-fine-tuning",
    n_saved=2,
    create_dir=True,
)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {"models": model})

# %% [markdown] Cell 246
# ### Model Training
#
# We decided to run the training phase for just one epoch because
# 1. each epoch is time-demanding when it ran over the entire dataset ($\approx 650.000$ samples),
# 2. the model's improvement after consecutive epochs is null, e.g., the accuracy does not increase with two epochs of training,
# 3. we reached the performance documented in the paper "[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)" at first try.
#
# For these reasons, we assumed that DistilBERT would not remarkably improve after the first epoch, and that should be our time-horizon.

# %% [code] Cell 247
trainer.run(train_dataloader, max_epochs=NUM_EPOCHS)

# %% [markdown] Cell 248
# #### Performance Assessment
#
# Finally, we evaluate the model over the testing set.

# %% [code] Cell 249
tester = Engine(evaluate_step)
pbar_test = ProgressBar()
pbar_test.attach(tester, output_transform=lambda x: {"loss": x})
Accuracy().attach(tester, "accuracy")
ConfusionMatrix(num_classes=NUM_CLASSES).attach(tester, "cm")
Recall(average=True).attach(tester, "recall")
Precision(average=True).attach(tester, "precision")
tester.run(test_dataloader)
metrics = tester.state.metrics
avg_accuracy = metrics["accuracy"]

print(
    f"Testing Results - Avg accuracy: {avg_accuracy:.3f}\nTesting Results - Avg recall: {metrics['recall']}\nTesting Results - Avg precision: {metrics['precision']}"
)

# %% [code] Cell 250
ConfusionMatrixDisplay(
    confusion_matrix=metrics["cm"].numpy(), display_labels=range(1, NUM_CLASSES + 1)
).plot()
plt.show()

# %% [markdown] Cell 251
# ### Conclusions
#
# Some considerations about the results.
# * The transfer learning version of DistilBERT has an accuracy of **0.554** on the testing set, which is a bad result considering other models we've trained.
# * As in the DistilBERT model fine-tuned from the pre-trained model, the confusion matrix has the same diagonal pattern, but the values are lower (this is trivial since the accuracy is lower).
#
# Transfer learning is in general a powerful technique in machine learning tasks, but it is not applicable when using pre-trained models like DistilBERT. Pre-trained models are already trained on a large corpus of text, so they already are good at extracting features from text data, however, they are not specific of a task. So, the transfer learning approach is not useful in this case since there is no a priori knowledge to transfer from the pre-trained model to the layers of the classification head.

# %% [code] Cell 252
import seaborn as sns

# %% [code] Cell 253
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise SystemError("GPU device not found")

# %% [markdown] Cell 254
# ## Zero-Shot Classification
#

# %% [markdown] Cell 255
# In this section, we will evaluate the performance of zero-shot classification on the Yelp Review Full dataset. The following zero-shot approach is not based on generative LLMs but on the concept of a *universal classifier*, a classifier trained on a universal task, and a form of instruction or prompt enables it to generalize to unseen classification tasks. The model of the *universal classifier* used here is based on the concept of Natural Language Inference (NLI). The NLI task is defined as recognizing if the meaning of one text (the hypothesis) is entailed in another text (the premise). In more detail, given a premise and a hypothesis, a NLI model must be able to recognize if the hypothesis is entailed by the premise, if it's a contradiction, or give a neutral opinion when it's not clearly entailed or a contradiction. For simplicity, the neutral and contradiction cases are merged together, resulting a more universal task of entailment vs. non-entailment.
#
# This approach allows us to reformulate any text classification task as entailment vs. non-entailment through label verbalization. For example, in topic classification, the task could be to determine if the text _"The red panda inhabits coniferous forests as well as temperate broadleaf and mixed forests"_ belongs to the topic _"nature"_ or _"history"_. From an NLI perspective, we can interpret the input text as the premise and verbalize the topic labels into two topic hypotheses: _"This text is about nature"_ and _"This text is about history"_. Finally, the ultimate decision is to determine which of the two topic hypotheses is more consistent with the text of interest, which in practice, usually consists of choosing the hypothesis with the highest score.
#
# The main disadvantage of NLI for universal classification is that it requires a separate prediction for each of the N class hypotheses, creating computational overhead for tasks with many classes.
#
# ref:
# _[Moritz Laurer , Wouter van Atteveldt , Andreu Casas† , Kasper Welbers. 2024. Building Efficient Universal Classifiers with Natural Language Inference.](https://arxiv.org/abs/2312.17543)_

# %% [markdown] Cell 256
# We'd like to test how the model behaves when feeded with different verbalisation forms for the labels. We try three different verbalisation forms on a small subset of the training set that will be used as a validation set to determine which verbalisation is better and finally evaluate the performance of the model with the choosen verbalisation form on the entire testset, to have consistent metrics with the rest of the models we tried before.

# %% [markdown] Cell 257
# We proceed by downloading the model deberta-v3-base, specifically tuned for zeroshot NLI.

# %% [code] Cell 258
trainset = load_dataset("yelp_review_full", split="train[:2%]")

model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
zeroshot_classifier = pipeline(
    task="zero-shot-classification", model=model_name, device=device
)

# %% [markdown] Cell 259
# ### 1st label verbalisation form
#
# The first verbalisation format that we are going to test is "This review is rated {} out of 4", where {} will be determined by the model choosing from 5 possible candidates: the actual range of the review from 0 to 4.
#
# To make the prediction we make use of the `pipeline` module from HuggingFace, which allows us to streamline the entire inference phase of a transformer model, from preprocessing of the input text to generating the inference output in few lines of code.

# %% [code] Cell 260
# %%time  # (magic command commented out)
ratings = [0, 1, 2, 3, 4]

hypothesis_template = "This review is rated {} out of 4."
result = list()

for out in tqdm(
    zeroshot_classifier(
        KeyDataset(trainset, "text"),
        candidate_labels=ratings,
        hypothesis_template=hypothesis_template,
    )
):
    result.append(out)

# %% [code] Cell 261
# Initialize the list to store all predictions
candidate_labels = ratings
all_predictions = [pred["labels"][0] for pred in result]

# Calculate the accuracy using sklearn's accuracy_score
true_labels = trainset["label"]
accuracy = accuracy_score(true_labels, all_predictions)
print(f"Accuracy: {accuracy:.2%}")

# Create the confusion matrix
cm = confusion_matrix(true_labels, all_predictions, labels=candidate_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=candidate_labels,
    yticklabels=candidate_labels,
    annot=True,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [markdown] Cell 262
# We can observe that this kind of approach performs poorly, achieving even lower accuracy than the Logistic Regression baseline.

# %% [markdown] Cell 263
# --------------------------------------------------------------------------------------------

# %% [markdown] Cell 264
# ### 2nd label verbalisation form
#
# We move to the next verbalisation form: "This review can be considered {}.", where {} will be substituted by one of the candidate from `[very negative, negative, neutral, positive, very positive]`.

# %% [code] Cell 265
label_mapping_1 = {
    "very negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very positive": 4,
}


def sentiment_to_id(sentiments, sentiment2id):

    ids = []
    for sentiment in sentiments:
        ids.append(sentiment2id[sentiment])

    return ids


# %% [code] Cell 266
hypothesis_template = "This review can be considered {}."
result = list()

for out in tqdm(
    zeroshot_classifier(
        KeyDataset(trainset, "text"),
        candidate_labels=list(label_mapping_1.keys()),
        hypothesis_template=hypothesis_template,
    )
):
    result.append(out)

# %% [code] Cell 267
# Initialize the list to store all predictions
candidate_labels = ratings
all_predictions = sentiment_to_id(
    [pred["labels"][0] for pred in result], label_mapping_1
)

# Calculate the accuracy using sklearn's accuracy_score
true_labels = trainset["label"]
accuracy = accuracy_score(true_labels, all_predictions)
print(f"Accuracy: {accuracy:.2%}")

# Create the confusion matrix
cm = confusion_matrix(true_labels, all_predictions, labels=candidate_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=candidate_labels,
    yticklabels=candidate_labels,
    annot=True,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [markdown] Cell 268
# We can observe that the performance has improved with respect to the _"rating"_ verbalisation, but it still does not reach the results obtained by the simpler Logistic Regressor model. We can notice that the model tends to mispredict the `very negative`, `very positive`, and `neutral` labels, preferring to give either the `positive` or the `negative` label. This behavior can be explained by the fact that when predicting ratings, a sentiment analysis approach is preferred, focusing on distinguishing the two binary classes `positive` or `negative`. Indeed, as indicated by the paper, this version of DeBERTa has been fine-tuned on the Yelp review dataset, considering only the 0-star and 4-star reviews, mapping each to either `negative` or `positive`. Let's continue our experiment on the next verbalised format.
#

# %% [markdown] Cell 269
# -------------------------

# %% [markdown] Cell 270
# ### 3rd label verbalisation form
#
# In the next experiment, we try to enrich the vocabulary used to describe the reviews by using terms representing different degrees of like and dislike. Here we used `awful`, `bad`, `neutral`, `good`, `excellent` to describe the rating of the review from 0 to 4.

# %% [code] Cell 271
label_mapping_2 = {"awful": 0, "bad": 1, "neutral": 2, "good": 3, "excellent": 4}

hypothesis_template = "What is being reviewed can be better described as {}"
result = list()

for out in tqdm(
    zeroshot_classifier(
        KeyDataset(trainset, "text"),
        candidate_labels=list(label_mapping_2.keys()),
        hypothesis_template=hypothesis_template,
    )
):
    result.append(out)

# %% [code] Cell 272
# Initialize the list to store all predictions
candidate_labels = ratings
all_predictions = sentiment_to_id(
    [pred["labels"][0] for pred in result], label_mapping_2
)

# Calculate the accuracy using sklearn's accuracy_score
true_labels = trainset["label"]
accuracy = accuracy_score(true_labels, all_predictions)
print(f"Accuracy: {accuracy:.2%}")

# Create the confusion matrix
cm = confusion_matrix(true_labels, all_predictions, labels=candidate_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=candidate_labels,
    yticklabels=candidate_labels,
    annot=True,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [markdown] Cell 273
# We can observe that the newly introduced, richer vocabulary does not help improve performance. In fact, the performance decreases by 4% compared to the previous, simpler vocabulary. It's clear that the model remains conservative, preferring to give predictions that are either 'good' or 'bad' (a direct mapping of 'positive' and 'negative'). This can be interpreted as another clue that the model has been trained to perform binary sentiment analysis.

# %% [markdown] Cell 274
# ---

# %% [markdown] Cell 275
# ### No label verbalisation
#
# In this experiment, we aim to investigate how much the verbalisation format of the labels influences the results. We consider our best-performing model so far (the one using the `[very negative, negative, neutral, positive, very positive]` candidate labels) and apply the inference approach without verbalising the labels. Instead of using a well-formatted prompt, we feed the labels directly to the model, with the hypothesis clause composed of just the label.

# %% [code] Cell 276
result = list()
for out in tqdm(
    zeroshot_classifier(
        KeyDataset(trainset, "text"), candidate_labels=list(label_mapping_1.keys())
    )
):
    result.append(out)

# %% [code] Cell 277
# Initialize the list to store all predictions
candidate_labels = ratings
all_predictions = sentiment_to_id(
    [pred["labels"][0] for pred in result], label_mapping_1
)

# Calculate the accuracy using sklearn's accuracy_score
true_labels = trainset["label"]
accuracy = accuracy_score(true_labels, all_predictions)
print(f"Accuracy: {accuracy:.2%}")

# Create the confusion matrix
cm = confusion_matrix(true_labels, all_predictions, labels=candidate_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=candidate_labels,
    yticklabels=candidate_labels,
    annot=True,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [markdown] Cell 278
# The prediction behaviour remains the same (not unexpected), but the interesting part is the drop in accuracy, suggesting that crafting a good hypothesis clause can give a significant boost to performance for this type of approach.

# %% [markdown] Cell 279
# ### Model Evaluation on the test set
#
# We now proceed to evaluate the performance of the zero-shot deberta modle using the verbalisation format which previously yielded the best performance on the small training test used as valdidation.

# %% [code] Cell 280
testset = load_dataset("yelp_review_full", split="test")

# %% [code] Cell 281
# %%time  # (magic command commented out)

hypothesis_template = "This review can be considered {}."
result = list()

for out in tqdm(
    zeroshot_classifier(
        KeyDataset(testset, "text"),
        candidate_labels=list(label_mapping_1.keys()),
        hypothesis_template=hypothesis_template,
    )
):
    result.append(out)

# %% [code] Cell 282
# Initialize the list to store all predictions
candidate_labels = ratings
all_predictions = sentiment_to_id(
    [pred["labels"][0] for pred in result], label_mapping_1
)

# Calculate the accuracy using sklearn's accuracy_score
true_labels = testset["label"]
accuracy = accuracy_score(true_labels, all_predictions)
print(f"Accuracy: {accuracy:.2%}")

# Create the confusion matrix
cm = confusion_matrix(true_labels, all_predictions, labels=candidate_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=candidate_labels,
    yticklabels=candidate_labels,
    annot=True,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [markdown] Cell 283
# The performance dropped by a few percentage points compared to the validation test. The main takeaway from the zero-shot approach on our rating prediction task is that it is unable to achieve even the performance of a simple Logistic Regressor, while requiring much more computational power and time to make the predictions. Furthermore, our findings suggest that optimizing the verbalization format for candidate labels can potentially improve performance compared to directly inputting the labels as they are.

# %% [markdown] Cell 284
# ## Conclusions on 5-star rating prediction

# %% [code] Cell 285
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% [code] Cell 286
five_star_prediction_accuracy = {
    "Logistic Regression": 0.5770,
    "Charcter-level CNN": 0.6228,
    "Custom LSTM": 0.6039,
    "fine-tuned distilBERT": 0.6840,
    "transfer-learning distilBERT": 0.5540,
    "DeBERTa ZeroShot": 0.4755,
}

# %% [code] Cell 287
data = pd.DataFrame.from_dict(
    five_star_prediction_accuracy, orient="index", columns=["Accuracy"]
)
data = data.reset_index()
data = data.rename(columns={"index": "Model"})

# Create the bar plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Model", y="Accuracy", data=data)
ax.bar_label(ax.containers[0], fontsize=12)
ax.plot("fine-tuned distilBERT", 0.72, "*", markersize=20, color="r")

# Customize the plot
plt.title("5-star rating Prediction Accuracy", fontsize=16, fontweight="bold")
plt.xlabel("Model", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0.4, 0.8)
# Adjust the spacing
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)

# Display the plot
plt.show()


# %% [markdown] Cell 288
# **Conclusions**\
# The results obtained from different approaches highlight the challenges and complexities involved in this task due to the inherent subjectivity in associating textual content, written by different subjects, with specific ratings.
#
# Model Performance
#
#   - Logistic Regression: Achieved an accuracy of 57.7%. This baseline model performed reasonably well considering its simplicity but struggled to capture the nuances of the text data, especially among close rating values.
#
#   - Character-level CNN: Achieved an accuracy of 62.3%. This model improved over logistic regression by learning features directly at the character level, thus capturing more granular patterns in the text.
#
#   - Custom LSTM Network: Achieved an accuracy of 60.4%. While LSTM networks are adept at handling sequential data, our custom implementation did not outperform the character-level CNN, likely due to the complexity and the similarity of sequences of different ratings.
#
#   - Fine-tuned DistilBERT: Achieved the highest accuracy of 68.4%. Fine-tuning DistilBERT, a distilled version of BERT, yielded the best results, demonstrating the power of pre-trained transformer models when fine tuned on the specific task on interest, in understanding and classifying textual data. The significant improvement underscores the importance of using advanced pre-trained models and also fine-tuning them on the specific task.
#
#   - DistilBERT with Transfer Learning: Achieved an accuracy of 55.4%. Freezing most of the DistilBERT model and only training the final layers led to a significant drop in performance, indicating that fine-tuning the entire model is crucial for better capturing the complexity of the task and of the specific Yelp dataset.
#
#   - DeBERTa ZeroShot: Achieved an accuracy of 47.6%. The zero-shot approach with DeBERTa did not perform well, highlighting the limitations of applying models without task-specific fine-tuning for this context.
#
# We conclude that is difficult to surpass the DistilBERT results, given the subjectivity and the complexity of the task.
#
# In fact we have:
# - Subjectivity in Ratings: The task of associating a review with a specific rating is inherently subjective. Different reviewers may have different interpretations of the same review, leading to variability in the labeled data.
# - Human-Level Difficulty: Even for humans, categorizing reviews into precise ratings can be challenging due to the nuanced and often subjective nature of personal opinions and experiences shared in the reviews.
# - Model Limitations: Despite leveraging advanced models, achieving an accuracy higher than 0.684 proved difficult. This ceiling suggests that the models' performance may be constrained by the subjective nature of the task and the variability in the data.
#
# Given the subtle nuances between adjacent star ratings and the challenges associated with accurately classifying reviews into five distinct classes, we propose two extensions to the text classification task to explore different granularities and simplify the complexity:
# - Three-Class Star Rating Classification
# - Polarity Star Rating Classification
#

# %% [markdown] Cell 289
# # Extensions:

# %% [markdown] Cell 290
# To extend our task and get more insights on the problem we decided to perform the following extensions:
#
# * investigate sligthly different tasks on the same dataset to better understand the data and the problem:
#   * sentiment analysis on three classes, considering {0, 1} negative sentiment, {2} as neutral class and {3, 4} as positive sentiment
#   * sentiment analysis on a polarity Yelp dataset, considering the ratings {0, 1, 2} as negative sentiment and {3, 4} as positive sentiment (we also tried a version with {0,1} negative sentiment and {3,4} positive sentiment)
#
# * investigate the same task on another dataset, the DBpedia, mentioned in the considered paper

# %% [markdown] Cell 291
# ## Imports

# %% [code] Cell 292
# !pip install datasets  # (magic command commented out)
# !pip3 install datasets contractions nltk gensim  # (magic command commented out)
# !pip3 install torch  # (magic command commented out)
# !pip3 install -q transformers datasets  # (magic command commented out)
# !pip3 install --upgrade scikit-learn==1.0.2  # (magic command commented out)
# !pip3 install matplotlib  # (magic command commented out)
# !pip3 install accelerate -U  # (magic command commented out)

# %% [code] Cell 293
import os
import random
import string
import subprocess
import sys
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Fix randomness and hide warnings
seed = int(os.getenv("PYTHONHASHSEED", 42))

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

np.random.seed(seed)
random.seed(seed)

# %% [code] Cell 294
import re
import string
import warnings
from collections import Counter
from string import punctuation

import contractions
import numpy as np
import pandas as pd
from datasets import load_dataset

# import gensim.downloader as api

# ignore annoying warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

# %% [code] Cell 295
import nltk

nltk.download("stopwords")
nltk.download("punkt")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stopwords = set(stopwords.words("english"))

# %% [code] Cell 296
import tensorflow as tf

# %% [code] Cell 297
from datasets import Dataset, DatasetDict, KeyDataset
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# %% [markdown] Cell 298
# ## Investigate a slightly different task on the same dataset

# %% [markdown] Cell 299
# ### Yelp dataset with 3 labels
#
# Given the model's tendency to mispredict ratings that are semantically close (such as predicting a 4 as a 3, or a 0 as a 1 and viceversa), we conduct another experiment where the rating 3-4 are merged into a single class representing positive sentiments, and the rating 0-1 into another class representing negative sentiments.

# %% [code] Cell 300
dataset = load_dataset("yelp_review_full", split="train")
testset = load_dataset("yelp_review_full", split="test")

# %% [code] Cell 301
dataset_df = pd.DataFrame(dataset)
testset_df = pd.DataFrame(testset)

# %% [markdown] Cell 302
# #### Logisitc Regression


# %% [code] Cell 303
def downcast_to_3_labels(x):
    if x < 2:
        return 0
    elif x > 2:
        return 2
    else:
        return 1


train = dataset_df["text"]
label = dataset_df["label"].map(downcast_to_3_labels)

# %% [code] Cell 304
vectorizer = CountVectorizer(min_df=50, stop_words="english", lowercase=True)
vectorizer.fit(train)

print("Vocabulary size: ", len(vectorizer.get_feature_names_out()))

# %% [code] Cell 305
train_x_vector = vectorizer.transform(train)
train_x_vector

# %% [code] Cell 306
label.value_counts()

# %% [markdown] Cell 307
# When performing the aggregation of the 3-4 and 0-1 ratings we end up with an unbalanced dataset, with the 1-star rating samples having half of the samples compared to the other two classes. This imbalance can induce the model to be biased towards the majority class, leading to poor performance on the minority class.
#
# To mitigate this issue we adopt the approach of assigning different weights to the classes in the loss function, to adjust the importance of each class, making the model more sensitive to the minority class and less biased towards the majority class.
#
# We opted to compute the weight for each class as the **Inverse Class Frequency**.

# %% [code] Cell 308
n_samples_for_class = dict(label.value_counts())

n_samples = label.shape[0]
n_classes = len(n_samples_for_class)

class_weight = dict()
for cls in n_samples_for_class:
    class_weight[cls] = n_samples / (n_classes * n_samples_for_class[cls])

print(class_weight)

# %% [code] Cell 309
model = LogisticRegression(class_weight=class_weight).fit(train_x_vector, label)

# %% [code] Cell 310
test = testset_df["text"]
test_label = testset_df["label"].map(downcast_to_3_labels)

# %% [code] Cell 311
test_vector = vectorizer.transform(test)
predictions = model.predict(test_vector)

print("Model accuracy: ", accuracy_score(predictions, test_label))
print("\nClassification report:\n")
print(classification_report(test_label, predictions, target_names=["0", "1", "2"]))

# %% [code] Cell 312
cm = confusion_matrix(test_label, predictions)


plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=["0", "1", "2"],
    yticklabels=["0", "1", "2"],
    annot=True,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [markdown] Cell 313
# #### Custom LSTM
#

# %% [markdown] Cell 314
# In this section we implement a new model which structure is completely equal to the LSTM shown before, but that has 3 classes in the output layer instead of 5. The purpose of the analysis is to build a model that recognizes the polarity of the review, understanding if a text is positive (5 and 4 stars review), neutral (3 stars), or negative (2 and 1 stars).

# %% [markdown] Cell 315
# First of all: the setup, which is exactly the same as before in terms of text preprocessing and embedding layer building.

# %% [code] Cell 316
# length of the vector representing the review
max_sentence_length = 300
# if the review is shorter, we add zeros in the end
padding = "post"
# if the review is longer, we truncate the words over the 300th
truncating = "post"

# activation function used for the output layer of the neural network
activation = "softmax"
# loss function used by the neural network
loss = "categorical_crossentropy"


# function used to remove the punctuation from the text, substituting it with a white space
def remove_punctuation(text):
    punct_regex = "[" + string.punctuation + "]"
    return re.sub(punct_regex, " ", text)


# preprocessing function of a text
def remove_noise(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = remove_punctuation(text)
    # remove contractions (e.g., That's become That is)
    text = contractions.fix(text)
    # list of words in the text
    tokens = nltk.word_tokenize(text)
    # remove stopwords
    tokens = [word for word in tokens if word not in stopwords]
    # build the text back
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


# function used to convert the text to a sequence of defined length, applies padding and truncating
def sequence_padding(sentences, tokenizer):
    sequences = tokenizer.texts_to_sequences(sentences)
    pad_trunc_sequences = sequence.pad_sequences(
        sequences, maxlen=max_sentence_length, padding=padding, truncating=truncating
    )

    return pad_trunc_sequences


# function used to map the labels from the instances in the dataset for a 3 class classification
def map_label(label):
    # 5 and 4 stars reviews are mapped to value 2
    if label > 2:
        return 2
    # 3 stars reviews are mapped to value 1
    elif label == 2:
        return 1
    # 2 and 1 stars reviews are mapped to value 0
    else:
        return 0


# function used to create the neural network's model
def create_model(vocab_size, emb_matrix, output_size):
    model = Sequential()

    # define input shape
    model.add(Input(shape=(max_sentence_length,)))

    # build embedding layer with defined parameters
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embed_vector_len,
        input_length=max_sentence_length,
        trainable=False,
    )
    embedding_layer.build((1,))
    # set the weights of the embedding layer to the values returned by the GloVe model
    embedding_layer.set_weights([emb_matrix])
    model.add(embedding_layer)

    model.add(Bidirectional(LSTM(256)))

    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))

    model.add(Dense(output_size, activation=activation))

    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

    return model


# %% [code] Cell 317
embed_vector_len = 300

glove_model = api.load(f"glove-wiki-gigaword-{embed_vector_len}")

# %% [markdown] Cell 318
# We build now the same model as per 5 stars classification, but the output layer will have size of 3, and not 5.

# %% [code] Cell 319
trainset_df = pd.DataFrame(dataset)
testset_df = pd.DataFrame(testset)

# %% [code] Cell 320
trainset_df["text"] = trainset_df["text"].apply(remove_noise)
testset_df["text"] = testset_df["text"].apply(remove_noise)

# %% [code] Cell 321
features = trainset_df["text"]

tokenizer = Tokenizer(oov_token="UNK")
tokenizer.fit_on_texts(features)
word_index = tokenizer.word_index

# %% [code] Cell 322
word_frequencies = Counter(word for text in features for word in text.split())
min_word_frequency = 50
common_words = [
    word for word, freq in word_frequencies.items() if freq >= min_word_frequency
]

word_index_filtered = {word: index for index, word in enumerate(common_words, start=1)}
vocab_size_filtered = len(word_index_filtered) + 1

tokenizer.word_index = word_index_filtered
tokenizer.word_index[tokenizer.oov_token] = 0

# %% [code] Cell 323
emb_matrix = []

# this array of zeros represents the embedding vector for the unknown character
emb_matrix.append(np.zeros(embed_vector_len))

# for each couple in the items of the (new) word_index of the tokenizer
for word, index in tokenizer.word_index.items():
    # if the word is recognized by the GloVe model
    if word in glove_model:
        # we get its embedding vector and append it to the embedding matrix
        embedding_vector = glove_model.get_vector(word)
        emb_matrix.append(embedding_vector)
    # if it is not unkown, we insert an array of zeros
    elif word != "UNK":
        emb_matrix.append(np.zeros(embed_vector_len))

emb = np.array(emb_matrix)

# %% [code] Cell 324
train_x = trainset_df["text"]
train_y = trainset_df["label"]
test_x = testset_df["text"]
test_y = testset_df["label"]

# %% [code] Cell 325
train_x = sequence_padding(train_x, tokenizer)
test_x = sequence_padding(test_x, tokenizer)

# %% [code] Cell 326
labels = [0, 1, 2]

# %% [markdown] Cell 327
# From the preprocessing point of view nothing changes for the texts, while the labels of the review need to be mapped to the new domain {0, 1, 2}.

# %% [code] Cell 328
train_y = train_y.apply(map_label)
test_y = test_y.apply(map_label)

# %% [markdown] Cell 329
# Now we build the new vector for the one-hot encoding using again the to_categorical function.

# %% [code] Cell 330
train_oh_y = to_categorical(train_y, num_classes=len(labels))
test_oh_y = to_categorical(test_y, num_classes=len(labels))

# %% [markdown] Cell 331
# We can build the new model, using the same function as before, except for the last parameter, now set to 3.

# %% [code] Cell 332
model_3 = create_model(vocab_size_filtered, emb, 3)
model_3.summary()

# %% [code] Cell 333
callbacks_3 = [
    EarlyStopping(monitor="val_loss", patience=1),
    ModelCheckpoint("model_3.keras", save_best_only=True, save_weights_only=False),
    ReduceLROnPlateau(patience=1),
]

# %% [code] Cell 334
history_3 = model_3.fit(
    train_x,
    train_oh_y,
    epochs=10,
    batch_size=32,
    verbose=1,
    validation_split=0.2,
    callbacks=callbacks_3,
)

# %% [markdown] Cell 335
# We evaluate the performance of the new model for the task of 3-class classification.

# %% [code] Cell 336
train_loss, train_accuracy = model_3.evaluate(train_x, train_oh_y)

print("Final Performance on the Training Set:")
print("\tAccuracy:", train_accuracy)
print("\tLoss:", train_loss)
print("\n")

test_loss, test_accuracy = model_3.evaluate(test_x, test_oh_y)

print("Performance on the Test Set:")
print("\tAccuracy:", test_accuracy)
print("\tLoss:", test_loss)

# %% [markdown] Cell 337
# ##### Conclusions

# %% [markdown] Cell 338
# As shown by the confusion matrix below, the cells that dominate are the correct prediction for positive and negative reviews. The problems are with the cells around (1, 1). In fact, we see that the model struggles to understand when to state that a review is neutral, since as we see there are like 6000 instances that were predicted as negative or positive, but were neutral. On the other way around, when it predicts neutral it is right 2/3 of the times.

# %% [code] Cell 339
predictions = model_3.predict(test_x)

# %% [code] Cell 340
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(test_y, predictions.argmax(axis=1))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm.T, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=True)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")
plt.show()

# del model_3

# %% [markdown] Cell 341
# #### ZeroShot 3-labels

# %% [markdown] Cell 342
#
# Since the DeBERTa model by Moritz Laurer has been fine-tuned on positive-negative sentiment predictions, we found it intersting to conduct an additional experiment with zero-shot classification and assess whether it enhances the performance in this context

# %% [code] Cell 343
model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
zeroshot_classifier = pipeline(
    task="zero-shot-classification", model=model_name, device=device
)

tri_label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
hypothesis_template = "This review can be considered {}."

result = list()
for out in tqdm(
    zeroshot_classifier(
        KeyDataset(testset, "text"),
        candidate_labels=list(tri_label_mapping.keys()),
        hypothesis_template=hypothesis_template,
    )
):
    result.append(out)


# %% [code] Cell 344
def downcast_to_3_labels(x):
    if x < 2:
        return 0
    elif x > 2:
        return 2
    else:
        return 1


tri_labels = testset["label"]
tri_labels = list(map(lambda x: downcast_to_3_labels(x), tri_labels))

# %% [code] Cell 345
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Initialize the list to store all predictions
candidate_labels = [0, 1, 2]
all_predictions = sentiment_to_id(
    [pred["labels"][0] for pred in result], tri_label_mapping
)

# Calculate the accuracy using sklearn's accuracy_score
true_labels = tri_labels
accuracy = accuracy_score(true_labels, all_predictions)
print(f"Accuracy: {accuracy:.2%}")

# Create the confusion matrix
cm = confusion_matrix(true_labels, all_predictions, labels=candidate_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=candidate_labels,
    yticklabels=candidate_labels,
    annot=True,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [markdown] Cell 346
# In this case, the zero-shot classification approach reaches almost the same performance as the custom lstm on the task, it is only slightly better, and we can associate this result to the fact that the zero shot model used was trained only on positive and negative sentiment, so it works better in distinguishing less classes. The improvement compared to the 5-star rating prediction task is notable, indicating that zero-shot classification performs better when the task has been previously encountered during training.
#

# %% [markdown] Cell 347
# ### Polarity on Yelp dataset
#
# As suggested by the paper, we decided to explore a polarity classification task using the Yelp Review Full dataset. This involved modifying the dataset to create a binary classification problem, referred to as the Polarity Yelp Review task.
#
#  The referenced paper proposed categorizing reviews with 1 and 2 stars as negative and those with 3 and 4 stars as positive, but after some trials, we extended the definition of negative reviews to include those with 0 stars, considering in this way all the original dataset. Our modification aimed to remain as much as possible consistent to the nature of the dataset and the underlying sentiment classification problem, ensuring that both strong and moderate negative sentiments were adequately represented.
#
#  Incorporating the 0-star reviews as negative, though, introduced a class imbalance. To address this, we implemented class weights in our model training process, ensuring that the imbalance did not introduce bias towards a class affecting the model's performance.

# %% [code] Cell 348
dataset = load_dataset("yelp_review_full", split="train")
testset = load_dataset("yelp_review_full", split="test")

# %% [code] Cell 349
dataset_df = pd.DataFrame(dataset)

dataset_df

# %% [code] Cell 350
testset_df = pd.DataFrame(testset)

testset_df

# %% [markdown] Cell 351
# With the following function we modify the original dataset, specifically we modify the labels, putting to '0' (negative sentiment) the ratings between 0 and 2, and putting to '1' (positive sentiment) the ratings 3 and 4. In this way we transform the Yelp review dataset to perform the polarity text-classification. In the paper was suggested to perform this task removing the rating '0' to have a balanced dataset, but we obtained better results on this variant of the task, so we decided to keep all the data we had originally.


# %% [code] Cell 352
def create_polarity_labels(dataset):
    polarity_dataset = pd.DataFrame()
    new_label = []
    negative_labels = [0, 1, 2]

    for i in range(0, dataset.shape[0]):
        if dataset["label"][i] in negative_labels:
            new_label.append(0)
        else:
            new_label.append(1)

    polarity_dataset["text"] = dataset["text"]
    polarity_dataset["new_label"] = new_label
    return polarity_dataset


# %% [code] Cell 353
polarity_dataset = create_polarity_labels(dataset_df)
polarity_dataset["new_label"].value_counts()

# %% [code] Cell 354
# Remove punctuation
regex = "[" + string.punctuation + "]"
polarity_dataset["text"] = polarity_dataset["text"].str.replace(
    pat=regex, repl="", regex=True
)

# %% [code] Cell 355
polarity_dataset

# %% [markdown] Cell 356
# #### Logistic Regression

# %% [code] Cell 357
train = [txt for txt in polarity_dataset["text"]]
label = [lbl for lbl in polarity_dataset["new_label"]]

# %% [code] Cell 358
vectorizer = CountVectorizer(min_df=50, stop_words="english", lowercase=True)
vectorizer.fit(train)

print("Vocabulary size: ", len(vectorizer.get_feature_names_out()))

# %% [code] Cell 359
train_x_vector = vectorizer.transform(train)
train_x_vector

# %% [code] Cell 360
n_samples_for_class = dict(polarity_dataset["new_label"].value_counts())

n_samples = dataset_df.shape[0]
n_classes = len(n_samples_for_class)

class_weight = dict()
for cls in n_samples_for_class:
    class_weight[cls] = n_samples / (n_classes * n_samples_for_class[cls])

print(class_weight)

# %% [code] Cell 361
model = LogisticRegression(class_weight=class_weight).fit(train_x_vector, label)

# %% [code] Cell 362
polarity_testset = create_polarity_labels(testset_df)

polarity_testset["text"] = polarity_testset["text"].str.replace(
    pat=regex, repl="", regex=True
)
polarity_testset["new_label"].value_counts()

# %% [code] Cell 363
test = [txt for txt in polarity_testset["text"]]
test_label = [lbl for lbl in polarity_testset["new_label"]]

# %% [code] Cell 364
test_vector = vectorizer.transform(test)
predictions = model.predict(test_vector)

print("Model accuracy: ", accuracy_score(predictions, test_label))
print("\nClassification report:\n")
print(classification_report(test_label, predictions, target_names=["0", "1"]))

# %% [code] Cell 365
cm = confusion_matrix(test_label, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# %% [markdown] Cell 366
# By transforming the task into a polarity text classification and applying BoW and logistic regression, we notice that the model achieves better results. It achieves an accuracy of 86% and from the classification report we can notice that the model is effective in distinguishing between negative and positive sentiments, having a good precision and recall on both classes. Furthermore, we can also see from the confusion matrix that the number of misclassifications is relatively balanced, so we can say that the model doesn't have a significant bias towards one class over the other.
#
# The relatively small number of misclassifications suggests that there may be some overlap or ambiguity in the sentiment expressed in the reviews near the boundary between negative and positive classifications. This is expected in text data, where the sentiment can be subjective. As seen from the classification on the 5 classes, we can imagine that the misclassifications are primarly between classes that are near one to each other in the spectrum of possible values from 0 to 4, so even if the problem is remapped we can say that the negative and the positive sentiments are less distinguishable, mainly, when the true original label is between '2' and '3' (in this case between '2' and '3' being the split line between the classification of negative and positive).
#
# With this simple yet more effective model we achieve good results, but we will try, as follows, to improve this baseline using a more complex model, the DistilBERT model.

# %% [markdown] Cell 367
# #### Distilbert on polarity problem

# %% [markdown] Cell 368
# Let's see if a complex model can improve the performance on the polarity task.

# %% [code] Cell 369
# Prepare train and test data
train = [txt for txt in polarity_dataset["text"]]
train_label = [lbl for lbl in polarity_dataset["new_label"]]

test = [txt for txt in polarity_testset["text"]]
test_label = [lbl for lbl in polarity_testset["new_label"]]

# %% [code] Cell 370
# Prepare the environment to run the model with the GPU
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise SystemError("GPU device not found")

# %% [code] Cell 371
model_name = "distilbert-base-uncased"

# %% [code] Cell 372
# Load the pretrained DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(
    device
)

# %% [code] Cell 373
# Load the tokenizer used for DistilBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %% [code] Cell 374
from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(
    train, train_label, test_size=0.2, random_state=2307
)
len(train_x), len(valid_x)

# %% [code] Cell 375
# Transform the data into the dataset of the HuggingFace API format
train_data = Dataset.from_list(
    [{"text": txt, "label": lbl} for txt, lbl in zip(train_x, train_y)]
)
valid_data = Dataset.from_list(
    [{"text": txt, "label": lbl} for txt, lbl in zip(valid_x, valid_y)]
)
test_data = Dataset.from_list(
    [{"text": txt, "label": lbl} for txt, lbl in zip(test, test_label)]
)


# %% [code] Cell 376
# Use the tokenizer to convert the input strings into sequences of tokens
def tokenize_function(example):
    return tokenizer(
        example["text"], padding=True, truncation=True, return_tensors="pt"
    )


# %% [code] Cell 377
data = DatasetDict()
data["train"] = train_data
data["validation"] = valid_data
data["test"] = test_data

# %% [code] Cell 378
# Tokenize all the data
tokenized_data = data.map(tokenize_function, batched=True)

# %% [code] Cell 379
print("Vocabulary size: ", len(tokenizer.vocab))

# %% [code] Cell 380
print(train[0])
print("\n")
print(tokenizer(train[0]).input_ids)
print("\n")
print(tokenizer(train[0]).attention_mask)

# %% [code] Cell 381
# The tokenizer is passed to DataCollatorWithPadding in order to be used to pad the sequences
# The purpose is to ensure that during training or inference, all input sequences in batch are of same length
# This step is important for efficient batch processing, so we add it to better manage the training model
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% [code] Cell 382
# The function defines the set of metrics, than passed to the model to evaluate the performance of the classification
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


# %% [code] Cell 383
# Import the os module which provides a way of using operating system dependent functionality
import os

# %% [code] Cell 384
# Login to Hugging Face using the API token, to load the obtained model and the metrics
from huggingface_hub import login

login(token=os.environ.get("HF_TOKEN"))

# %% [markdown] Cell 385
# The following code sets up the training environment for a model using the Hugging Face transformers library, configuring training parameters and integrating the training and evaluation datasets, tokenizer, and the metrics computation function. The model will then be trained and evaluated using the `trainer` instance.

# %% [code] Cell 386
from transformers import Trainer, TrainingArguments

# Specify the repository where the model and the checkpoints will be saved
repo_name = "distilbert-on-polarity-yelp-reviews"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=metrics,
)

# %% [code] Cell 387
trainer.train()

# %% [code] Cell 388
trainer.evaluate()

# %% [code] Cell 389
# Push the trained model and its configuration to the Hugging Face Model Hub
trainer.push_to_hub()

# %% [code] Cell 390
# Use the model for predictions and evaluate the model
preds = trainer.predict(tokenized_data["test"])
y_pred = torch.argmax(torch.tensor(preds.predictions), dim=1).numpy()

# %% [code] Cell 391
print("Results for the DistilBERT model:")
print(f"accuracy: {accuracy_score(y_pred, test_label)}")
print(classification_report(test_label, y_pred, target_names=["0", "1"]))

# %% [code] Cell 392
cm = confusion_matrix(test_label, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# %% [markdown] Cell 393
# The DistilBERT model shows an overall accuracy of 91%, which is an improvement over the 87% accuracy of the logistic regression model. The F1-scores for both classes are higher with the DistilBERT model (0.93 and 0.89) compared to the logistic regression model (0.89 and 0.83). This indicates better precision and recall balance. The transformer-based architecture of DistilBERT leverages contextual understanding, making it more effective for sentiment analysis tasks than the traditional bag-of-words approach used in logistic regression, but as we have seen for the 5 classes it struggles in case of many options from which to choose when dealing with a subjective context, that could be associated to more than one label. Given this conclusion it would be interesting to perform on the same dataset multi-label classification, if we had some domain experts to better label our dataset, in order to then get better insights about the dataset and about the different models capabilities on this extended task in comparison with the original one.
#
# Finally we can say that on the polarity classification task the comparison between logistic regression and DistilBERT highlights the advantage of using advanced models like Transformers for text classification tasks, demonstrating their ability to provide more accurate and reliable predictions. In any case, the results will always depend also on the quality of the dataset and its objectiveness.

# %% [markdown] Cell 394
# #### Removing 2-star ratings

# %% [markdown] Cell 395
# Given that the model has high percentage of misprediction for the 2-star ratings, maybe "riconducibile" to the subjectivity of a neutral reaction and thus introducing noise and misleading reviews for the model, we'd like to conduct another experiment where we exclude these labels from training and inspect the effect of it's absence.
#
# In fact as noticed before, given that the negative and the positive sentiments are less distinguishable, mainly, when the true original label is between '2' and '3', at the split line between the two sentiments, we try to analyse the problem without the neutral label of '2' star rating, moving away from each other the values on the splitting line between the two classes. Let's see how the simple model of Logistic Regression performs in this situation.

# %% [code] Cell 396
index = dataset_df[~(dataset_df["label"] == 2)].index
len(index)

# %% [code] Cell 397
index = dataset_df[~(dataset_df["label"] == 2)].index

train = dataset_df["text"][index]
label = dataset_df["label"][index].map(lambda x: 1 if x > 2 else 0)

print(train.shape)
print(label.value_counts())

# %% [code] Cell 398
vectorizer = CountVectorizer(min_df=50, stop_words="english", lowercase=True)
vectorizer.fit(train)

print("Vocabulary size: ", len(vectorizer.get_feature_names_out()))

# %% [code] Cell 399
train_x_vector = vectorizer.transform(train)
train_x_vector

# %% [code] Cell 400
model = LogisticRegression().fit(train_x_vector, label)

# %% [code] Cell 401
index = testset_df[~(testset_df["label"] == 2)].index

test = testset_df["text"][index]
test_label = testset_df["label"][index].map(lambda x: 1 if x > 2 else 0)

print(test.shape)
print(test_label.value_counts())

# %% [code] Cell 402
test_vector = vectorizer.transform(test)
predictions = model.predict(test_vector)

print("Model accuracy: ", accuracy_score(predictions, test_label))
print("\nClassification report:\n")
print(classification_report(test_label, predictions, target_names=["0", "1"]))

# %% [code] Cell 403
cm = confusion_matrix(test_label, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"], annot=True
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [markdown] Cell 404
# #### Zero-shot 2-labels

# %% [markdown] Cell 405
# We try a more complex model without the neutral reviews, which could be ambiguos and misleading. We chose to try the zero shot to test how much it can improve when used on exactly two classes, giving the fact that it was originally trained and fine-tuned on positive-negative sentiment predictions.

# %% [code] Cell 406
testset_df = testset.to_pandas()
print(testset_df.shape)


# %% [code] Cell 407
def polarity_lbl(x):
    if x > 2:
        return 1
    elif x < 2:
        return 0
    elif x == 2:
        return None
    else:
        raise Exception(
            "Something went wrong when translating numeric to verbal labels"
        )


testset_df["polarity_label"] = testset_df["label"].map(polarity_lbl)
testset_df = testset_df[~testset_df["polarity_label"].isnull()]

print(testset_df.shape)

# %% [code] Cell 408
from datasets import Dataset

label_mapping_polarity = {"negative": 0, "positive": 1}

testset2 = Dataset.from_pandas(testset_df)

hypothesis_template = "This review can be considered {}."

result = list()
for out in tqdm(
    zeroshot_classifier(
        KeyDataset(testset2, "text"),
        candidate_labels=list(label_mapping_polarity.keys()),
        hypothesis_template=hypothesis_template,
    )
):
    result.append(out)

# %% [code] Cell 409
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Initialize the list to store all predictions
candidate_labels = [0, 1]
all_predictions = sentiment_to_id(
    [pred["labels"][0] for pred in result], label_mapping_polarity
)

# Calculate the accuracy using sklearn's accuracy_score
true_labels = testset_df["polarity_label"]
accuracy = accuracy_score(true_labels, all_predictions)
print(f"Accuracy: {accuracy:.2%}")

# Create the confusion matrix
cm = confusion_matrix(true_labels, all_predictions, labels=candidate_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=candidate_labels,
    yticklabels=candidate_labels,
    annot=True,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [markdown] Cell 410
# ## Conclusions

# %% [code] Cell 411
three_star_prediction_accuracy = {
    "Logistic Regression": 0.753,
    "Custom LSTM": 0.798,
    "DeBERTa ZeroShot": 0.8,
}

polarity_prediction_accuracy = {
    "Logistic Regression": 0.86,
    "fine-tuned distilBERT": 0.914,
}

# %% [code] Cell 412
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.DataFrame.from_dict(
    three_star_prediction_accuracy, orient="index", columns=["Accuracy"]
)
data = data.reset_index()
data = data.rename(columns={"index": "Model"})

# Create two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Create the first plot
ax1 = sns.barplot(x="Model", y="Accuracy", data=data, ax=ax1)
ax1.bar_label(ax1.containers[0], fontsize=12)
ax1.set_title("3-Star Rating Prediction Accuracy", fontsize=16, fontweight="bold")
ax1.set_xlabel("Model", fontsize=14)
ax1.set_ylabel("Accuracy", fontsize=14)
ax1.set_xticks(
    range(len(data["Model"])), data["Model"], rotation=45, ha="right", fontsize=12
)
ax1.set_yticks(range(len(data["Accuracy"])), data["Accuracy"], fontsize=12)
ax1.set_ylim(0.5, 1.0)

data = pd.DataFrame.from_dict(
    polarity_prediction_accuracy, orient="index", columns=["Accuracy"]
)
data = data.reset_index()
data = data.rename(columns={"index": "Model"})

# Create the second plot (replace this with your second dataset)
ax2 = sns.barplot(x="Model", y="Accuracy", data=data, ax=ax2)
ax2.bar_label(ax2.containers[0], fontsize=12)
ax2.set_title("Polarity prediction Accuracy", fontsize=16, fontweight="bold")
ax2.set_xlabel("Model", fontsize=14)
ax2.set_ylabel("Accuracy", fontsize=14)
ax2.set_xticks(
    range(len(data["Model"])), data["Model"], rotation=45, ha="right", fontsize=12
)
ax2.set_yticks(range(len(data["Accuracy"])), data["Accuracy"], fontsize=12)
ax2.set_ylim(0.5, 1.0)

# Adjust the spacing
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2, wspace=0.3)

# Display the plots
plt.show()

# %% [markdown] Cell 413
# **Conclusions**\
# Switching from a 5-class to a 3-class prediction model provides a notable boost in performance, which is expected, as previously most mispredictions occurred for ratings that were only one star apart. Since the decision to give a 4-star rather than a 3-star rating (and similarly for 1 or 0) can be highly subjective, aggregating 3-4 star rating into a positive class and 0-1 star ratings into a negative class helps mitigate the effect that this subjectivity plays in determing the rating of the review, and determing the general sentiment of a review becomes a much easier task.
#
# The greatest improvements can be seen for the zero-shot approach. Again, this can be explained by the fact that the model has been fine-tuned for sentiment analysis on positive-negative reviews, suggesting that if the task to be performed in the zero-shot classification is among the tasks for which the model has been trained, the zero-shot approach become a valid classification method.
#
# Regarding the neutral class (the reviews with 2-star ratings) we can see from the previous confusion matrix that the model has difficulty in correclty identifying this class, resulting in law recall and precision. This can be explained by the fact that neutral reviews most of the times result ambiguous, they tend to contain a mix of positive and negative statements, making it difficult for models to categorize them definitively.
#
# When we further simplified the task by aggregating neutral reviews into negative reviews (0-2 stars as negative, 3-4 stars as positive), we noticed a significant improvement in performance. This binary classification task reduced even more complexity and ambiguity, allowing models to better distinguish between the two classes. The success of the polarity task underscores the importance of reducing class ambiguity and highlights that models perform more effectively when tasked with clearer, more distinct categories. Following this intuition we tried to simplify the problem even more by removing the middle class, and we noticed an even further improvement in accuracy. This approach underscores the benefits of increasing the separation between the positive and negative classes, thereby simplifying the classification task and enhancing model performance.

# %% [markdown] Cell 414
# ## Investigate the same task on another related dataset

# %% [markdown] Cell 415
# We perform again multi-class classification, but on the DBpedia dataset, that in the paper was mentioned as one of the datasets on which were achieved the best results with the CharCNN model.
#
# The DBpedia classification dataset was constructed by picking 14 non-overlapping classes. From each one of these classes were randomly chosen 40000 training samples and 5000 testing samples. Therefore, the total size of the training dataset is 560000 trining samples and 70000 testing samples. There are 3 columns in the dataset, same for training and testing, corresponding to class index (from 0 to 13), the title and the content. We will drop the title and consider only the content and the class index for our task of text-classification, in which we want to predict the current topic given the topic.
#
# The dataset contains mainly English data, but words from other languages may appear, as in the Yelp reviews (ex. in DBpedia a film with a foreign title, while in Yelp a restaurant with foreign name); but regarding the stopwords we will consider only the English vocabulary, removing them before applying the BoW with the Logistic Regression model.
#
# The followed procedure is the same as for the Yelp dataset, but we achieve very different results.

# %% [code] Cell 416
related_dataset = load_dataset("fancyzhx/dbpedia_14", split="train")
related_testset = load_dataset("fancyzhx/dbpedia_14", split="test")

# %% [code] Cell 417
dataset_df_r = pd.DataFrame(related_dataset)
dataset_df_r = dataset_df_r.drop(columns="title")
dataset_df_r

# %% [code] Cell 418
testset_df_r = pd.DataFrame(related_testset)
testset_df_r = testset_df_r.drop(columns="title")
testset_df_r

# %% [code] Cell 419
train_x = [txt for txt in dataset_df_r["content"]]
label_x = [lbl for lbl in dataset_df_r["label"]]

# %% [code] Cell 420
vectorizer = CountVectorizer(min_df=50, stop_words="english", lowercase=True)
vectorizer.fit(train_x)

print("Vocabulary size: ", len(vectorizer.get_feature_names_out()))

# %% [code] Cell 421
vectorizer.get_feature_names_out()[:50]

# %% [code] Cell 422
train_x_vector = vectorizer.transform(train_x)

train_x_vector

# %% [code] Cell 423
model = LogisticRegression().fit(train_x_vector, label_x)

# %% [code] Cell 424
# Elements with highest positive coefficients, influencing the most the predictions
vocab = vectorizer.get_feature_names_out()
model_params = [(vocab[j], model.coef_[0][j]) for j in range(len(vocab))]
sorted(model_params, key=lambda x: -x[1])[:20]

# %% [code] Cell 425
vocab = vectorizer.get_feature_names_out()

for i, label in enumerate(set(label_x)):
    top10 = np.argsort(model.coef_[i])[-10:][::-1]
    if i == 0:
        top = pd.DataFrame(vocab[top10], columns=[label])
        top_indices = top10
    else:
        top[label] = vocab[top10]
        top_indices = np.concatenate((top_indices, top10), axis=None)

print(top)

# %% [markdown] Cell 426
# For the new dataset we can notice that the words with highest positive coefficients, influencing the most the predictions, are not adjectives, but nouns realted to different topics. In this case the task is related to topics, not ratings, so the vocabulary is more precise, less subjective and emotional, explaining a topic, without giving a personal opinion. We can also notice that there are no overlappings between the most important words used by the model to predict the specific class.
#
# Let's now see how much this influences the model predictions.

# %% [code] Cell 427
test_y = [txt for txt in testset_df_r["content"]]
label_y = [lbl for lbl in testset_df_r["label"]]

# %% [code] Cell 428
test_vector = vectorizer.transform(test_y)
predictions = model.predict(test_vector)

# %% [code] Cell 429
print("Model accuracy: ", accuracy_score(predictions, label_y))
print("\nClassification report:\n")
print(
    classification_report(
        label_y,
        predictions,
        target_names=[
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
        ],
    )
)

# %% [markdown] Cell 430
# We can see that the model performance is really high already, by using only a simple model as Logistic Regression with BoW. We can conclude that this is due to the different type of dataset, considering that we didn't changed anything in the applied procedure. The model works better when trying to identify the different topics, on an objective content, while it works much worse in case of predicting the ratings of reviews, which are subjective. In fact even in real life, for humans, is harder to agree on the rating, while we objectively classify the topics of a piece of text, having more confidence due to the use of distinguishable nouns, and not only adjectives that can be interpreted differently in the sentiment spectrum.
#
# Given the really good results obtained with the simple model, we decided to not apply more complex models, that require more computational resources and time, given that the result is acceptable and the model can be used, having a very good trade-off of performance and efficency.

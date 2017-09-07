from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# load the dataset and show the first 5 rows
# the table below shows an example.

#######################################
#   your code goes here               #
#######################################

import json
import pandas as pd
import numpy as np

filepath = '/Users/Miller/Documents/CSCI 3360/HW 3.5/amazon_data.json'

with open(filepath, 'r') as amazon_data:
    data = amazon_data.readlines()

amazon_list = []
for line in data:
    line.replace('\n','')
    amazon_list.append(json.loads(line))

amazon_df = pd.DataFrame(amazon_list)

amazon_df = amazon_df[['overall','reviewText']]

amazon_df['label'] = 0

amazon_df['label'] = np.where(amazon_df['overall']>=4,1,0)

stop_words = set(stopwords.words('english'))

# Making reviews lowercase
amazon_df.reviewText = amazon_df.reviewText.str.lower()

# Tokenizing reviews
amazon_df.reviewText = amazon_df.reviewText.apply(word_tokenize)

# Creating Porter Stemmer
PS = PorterStemmer()

amazon_df.review_filtered_stemmed = amazon_df.reviewText.apply(lambda x : [PS.stem(word) for word in x if (word not in stop_words) and (len(word) > 2)])

# Put each review into a string
reviews_as_strings = amazon_df.review_filtered_stemmed.apply(' '.join)

# Joining all reviews into one string
all_reviews = " ".join(reviews_as_strings)

# Creating word distribution
word_distribution = nltk.FreqDist(word_tokenize(all_reviews))

# Creating list of most frequent words
most_freq_words = word_distribution.most_common(2000)

# Removing punctuation from most_freq_words

punctuation = ["'", "."]

most_freq_words_with_punctuation = [word for word in most_freq_words for punc in punctuation if punc in word[0]]

most_freq_words_no_punctuation = list(set(most_freq_words) - set(most_freq_words_with_punctuation))

most_freq_words_with_punctuation

# Creating vocabulary (words only)
vocab = [i[0] for i in most_freq_words]

# len(most_freq_words_no_punctuation)

vocab = [i[0] for i in most_freq_words_no_punctuation]

count_vect = CountVectorizer(vocabulary=vocab)

X = count_vect.fit_transform(reviews_as_strings)

y = amazon_df.label

# number of examples to use for training
n_train = 30000

X_train, y_train = X[:n_train, :], y[:n_train]
X_test, y_test = X[n_train:, :], y[n_train:]


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


# construct and train naive bayes and logistic regression models, and then 
# print out their train/test accuracies

##################################
#  your code goes here           #
##################################

nb = MultinomialNB()
nb.fit(X_train,y_train)

nb_train_accuracy = nb.score(X_train,y_train)
nb_test_accuracy = nb.score(X_test,y_test)

print('Naive Bayes Train Accuracy: ' + str(nb_train_accuracy))
print('Naive Bayes Test Accuracy: ' + str(nb_test_accuracy))

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

log_reg_train_accuracy = log_reg.score(X_train,y_train)
log_reg_test_accuracy = log_reg.score(X_test,y_test)

print('Logistic Regression Train Accuracy: ' + str(log_reg_train_accuracy))
print('Logistic Regression Test Accuracy: ' + str(log_reg_test_accuracy))

#####################################################################


tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X_tf = tf_transformer.transform(X)

y = amazon_df.label

print("X.shape=", X_tf.shape)   # feature vectors (2d array)
print("y.shape=", y.shape)   # labels (1d array)


# number of examples to use for training
n_train = 30000

X_tf_train, y_train = X_tf[:n_train, :], y[:n_train]
X_tf_test, y_test = X_tf[n_train:, :], y[n_train:]

nb = MultinomialNB()
nb.fit(X_tf_train,y_train)

nb_train_accuracy = nb.score(X_tf_train,y_train)
nb_test_accuracy = nb.score(X_tf_test,y_test)

print('Naive Bayes Train Accuracy: ' + str(nb_train_accuracy))
print('Naive Bayes Test Accuracy: ' + str(nb_test_accuracy))

log_reg = LogisticRegression()
log_reg.fit(X_tf_train,y_train)

log_reg_train_accuracy = log_reg.score(X_tf_train,y_train)
log_reg_test_accuracy = log_reg.score(X_tf_test,y_test)

print('Logistic Regression Train Accuracy: ' + str(log_reg_train_accuracy))
print('Logistic Regression Test Accuracy: ' + str(log_reg_test_accuracy))

##########################

# Plots

#1

alphas = [1,5,10,15,20,25,30,35,40,45,50]
train_acc = []
test_acc =[]

for alpha_ in alphas:
    nb = MultinomialNB(alpha=alpha_)
    nb.fit(X_tf_train,y_train)
    train_acc.append(nb.score(X_tf_train,y_train))
    test_acc.append(nb.score(X_tf_test,y_test))

plt.plot(alphas,train_acc,label='Train Accuracy')
plt.plot(alphas,test_acc,label='Test Accuracy')
plt.xlabel('Alpha')
# plt.xticks(alphas)
plt.title('Accuracy vs. Alpha for Naive Bayes')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

#2

samples = [10000,15000,20000,25000,30000]
nb_train_acc = []
nb_test_acc =[]
lr_train_acc = []
lr_test_acc =[]

for sample in samples:
    
    # number of examples to use for training
    n_train = sample
    
    X_tf_train, y_train = X_tf[:n_train, :], y[:n_train]
    X_tf_test, y_test = X_tf[n_train:, :], y[n_train:]
    
    nb = MultinomialNB()
    nb.fit(X_tf_train,y_train)
    
    nb_train_acc.append(nb.score(X_tf_train,y_train))
    nb_test_acc.append(nb.score(X_tf_test,y_test))
    
    log_reg = LogisticRegression()
    log_reg.fit(X_tf_train,y_train)
    
    lr_train_acc.append(log_reg.score(X_tf_train,y_train))
    lr_test_acc.append(log_reg.score(X_tf_test,y_test))

# Plots Log. Reg. Accuracy
plt.plot(samples,lr_train_acc,label='Train Accuracy')
plt.plot(samples,lr_test_acc,label='Test Accuracy')
plt.xlabel('# of Samples')
plt.title('Accuracy vs. # of Training Samples (Log. Regression)')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

# Plots Naive Bayes Accuracy
plt.plot(samples,nb_train_acc,label='Train Accuracy')
plt.plot(samples,nb_test_acc,label='Test Accuracy')
plt.xlabel('# of Samples')
plt.title('Accuracy vs. # of Training Samples (Naive Bayes)')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

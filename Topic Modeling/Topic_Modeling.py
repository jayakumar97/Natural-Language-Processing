#!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from nltk.corpus import stopwords
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from nltk.stem import WordNetLemmatizer


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


# In[122]:


lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
print(stop_words)


# In[123]:


def clean_text(text):
    """
    Function to clean a incoming sentence
    """
    # Regex to remove punctuation and any number if possible
    temp = re.sub(r'\d+|[^\w\s]', '', text)
    lemmatized_words = []
    for word in temp.split():
        lemmatized_word = word.lower()
        if lemmatized_word in stop_words:
            continue
        lemmatized_word = lemmatizer.lemmatize(lemmatized_word)
        lemmatized_words.append(lemmatized_word)
    
    return ' '.join(word for word in lemmatized_words)


# In[124]:


def tfidfvectorizer(df):
    """
    Function to get a Set of string and perform TF-IDF
    """
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df)
    return vectorizer,tfidf


# In[125]:


def bagofwordvectorizer(df):
    """
    Function to get a Set of string and perform Bag of Words
    """
    vectorizer = CountVectorizer(min_df = 1)
    bow = vectorizer.fit_transform(df)
    return vectorizer,bow


# In[129]:


def lsa(vectorizer,train_data, number_of_components = -1):
    """
    If the number of components is given, the function straight away fits the model, other wise it will check the optimal 
    number of components based on computing the gradient value of the Explained_variance
    """
    if number_of_components == -1:
        explained_variances = []
        test = range(5, 12 , 1)
       
        # Calculate Explained variance for each combination of number of components
        for n in test:
            svd = TruncatedSVD(n_components=n)
            lsa = svd.fit(train_data)
            explained_variances.append(lsa.explained_variance_ratio_.sum())

        plt.figure(figsize=(20, 10))
        plt.plot(test, explained_variances)
        plt.title('Explained variance');
        number_of_components = test[np.abs(np.gradient(np.gradient(explained_variances))).argmax()]
    
    # Fit the model 
    lsa = TruncatedSVD(n_components=number_of_components, n_iter=100, random_state=50)
    lsa.fit_transform(train_data)
    
    # Print the topics with their terms
    terms = vectorizer.get_feature_names()
    topics={}
    for i, component in enumerate(lsa.components_):
        zipped = zip(terms, component)
        topic_terms=sorted(zipped, key = lambda t: t[1], reverse=True)[:20]
        topic_terms=list(dict(topic_terms).keys())
        print("Topic "+str(i)+": ",topic_terms)
        topics[str(i)] = topic_terms
    return lsa, topics


# In[127]:



# Loading the  Dataset
df=pd.read_csv(r'./Dataset/drugsComTrain_raw.csv')
print(df.head())

df = df['review'].to_numpy()
preprocessed_data = []
for i in range(len(df)):
    result = clean_text(df[i])
    preprocessed_data.append(result)

for i in range(5):
    print('Before preprocess:\n',df[i])
    print('After preprocess:\n',preprocessed_data[i])


# In[130]:


#Get tfidfvectorizer
lsavectorizer1, lsadata1 = tfidfvectorizer(preprocessed_data)
#Create LSA model with tfidf vector
lsatfidf,top_terms_tfidif=lsa(lsavectorizer1, lsadata1)

#Get Bagofword vectorizer
lsavectorizer2, lsadata2 = bagofwordvectorizer(preprocessed_data)
#Create LSA model with Bag of word vector
lsabow,top_terms_bow = lsa(lsavectorizer2, lsadata2) 


# In[62]:


# To print 10 Components with 20 words
lsatfidf,top_terms_tfidif=lsa(lsavectorizer1, lsadata1,10)
lsabow,top_terms_bow = lsa(lsavectorizer2, lsadata2,10) 


# In[78]:



def lda(vectorizeer, train_data):
    # Give input for grid Search params
    gridsearchparameters = {'n_components': [8],'doc_topic_prior': [0.01,0.1,0.5,1],'topic_word_prior': [0.01,0.1,0.5,1]}
    
    # Create LDA model and perform GridSearch
    lda = LatentDirichletAllocation(max_iter=5, learning_offset=50,random_state=50, n_jobs = -1)
    model = GridSearchCV(lda, param_grid=gridsearchparameters, verbose = 10)
    print("training start")
    model.fit(train_data)
    print("training ends")
    
    # This gives the best model based on the max loglikelihood value 
    best_mode = model.best_estimator_
    print("Best Params =", model.best_params_)
    print("Best Loglikelihood Score =", model.best_score_)
    
    return best_model


# In[116]:


# Load the Dataset
testdataset=pd.read_csv(r'./Dataset/drugsComTest_raw.csv')
print(testdataset.head())
drugs = testdataset['drugName'].to_numpy()
testdataset = testdataset['review'].to_numpy()
preprocessed_testdata = []
for i in range(len(testdataset)):
    result = clean_text(testdataset[i])
    preprocessed_testdata.append(result)


# In[80]:


# Get bag of word vectorizer for test dataset
vectorizer, data = bagofwordvectorizer(preprocessed_data)
#Create LDA model for bag of words
ldamodel = lda(vectorizer, data)


# In[81]:


#Get Tfidf vectorizer
tfidfvector, tfidfdata = tfidfvectorizer(preprocessed_data)
#Create LDA model for Tfidf
ldamodel2 = lda(tfidfvector, tfidfdata)


# In[91]:


def Findtopics(vectorizer, lda_model, n_words=20):
    featurewords = np.array(vectorizer.get_feature_names)
    topicwords = []
    for i,Component in enumerate(lda_model.components_):
        topicwordpos = zip(featurewords,Component)
        sortwords = sorted(topicwordpos, key= lambda x:x[1], reverse=True)[:n_words]
        print("Topic "+str(i)+" : ")
        print(sortwords)
    return topicwords

print("Topics for LDA with bagofwords ")
topicwords = show_topics(vectorizer=vectorizer, lda_model=ldamodel, n_words=20)

print("\n\nTopics for LDA with TFIDF")
topicwords = show_topics(vectorizer=tfidfvector, lda_model=ldamodel2, n_words=20)


# In[103]:



# Grouping the documents based on the review to group the drugs name
#Fit the model to get the topic contribution for the documents
doc_topic = ldamodel2.fit_transform(tfidfdata)


# In[119]:


print(doc_topic.shape)
topics = {}

for i in range(8):
    topics[i] = []

#Iterate through all the documents
for i,doc in enumerate(doc_topic):
    max = -1
    pos = -1
    #Find the maximum Topic contribution for that particular document
    for x,topic in enumerate(doc):
        if max < topic:
            max = topic
            pos = x
    # Add the index of the document to group it
    topics[pos].append(i)

#Sum should equals to the number of documents
sum =0
for i in range(8):
    sum= sum+len(topics[i])
    
#Get the group of drugs based on topic Zero
topiczero_drugs = []
for i in topics[0]:
    topiczero_drugs.append(drugs[i])
print(set(topiczero_drugs))



import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import spatial
from sklearn.cross_validation import train_test_split
import math
from sklearn.svm import SVR

#Function to perform necessary pre-processing on the string
def str_preprocessing(s):
    s = s.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    words_array = tokenizer.tokenize(s)
    s = ""
    for word in words_array :
        if word not in text.ENGLISH_STOP_WORDS:
             word = stemmer.stem(word)
        s = s + word + " "
    return s

#Used for calculting the occurences of last word in search query in product info
def str_common_word_count(str1, str2):
    words, cnt = str2.split(), 0
    for word in words:
        if (str1 == word):
            cnt+=1
    return cnt

#Used to check if the last word of the query occurs atleast once in the product info
def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

#Count the occurence of the entire string in the other string
def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

#Importing the Training, Testing, Product Description, Attribute Data Sets
train = pd.read_csv('H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/Home Depot Dataset/train.csv',encoding="ISO-8859-1")
test = pd.read_csv('H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/Home Depot Dataset/test.csv', encoding="ISO-8859-1")#[:1000] #update here
pro_desc = pd.read_csv('H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/Home Depot Dataset/product_descriptions.csv')#[:1000] #update here
attr = pd.read_csv('H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/Home Depot Dataset/attributes.csv')

#Extracting the Brand from the Attribute Dataset
brand = attr[attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

#Merging Product Descritpion and Brand with Training Data based on Product ID
train = pd.merge(train,pro_desc, how='left', on='product_uid')
train = pd.merge(train, brand, how='left', on='product_uid')

#Replacing all nan's in the train data with n/a, which will be removed later
for i in range(0,train.shape[0]):
    if(isinstance(train.iloc[i,6],str) == False):
        if(math.isnan(train.iloc[i,6]) == True):
            train.iloc[i,6] = 'n/a'
      
#Deleting null and missing values
train = train[train.brand != 'unbranded']
train = train[train.brand != 'Unbranded']
train = train[train.brand != '.N/A']
train = train[train.brand != 'n/a']

#Final size of the train data
train_size = train.shape[0]

#Cleaning the textual data by performing stemming, removal of punctuations, converting to lower case
train['search_term'] = train['search_term'].map(lambda x:str_preprocessing(x))
train['product_title'] = train['product_title'].map(lambda x:str_preprocessing(x))
train['product_description'] = train['product_description'].map(lambda x:str_preprocessing(x))
train['brand'] = train['brand'].map(lambda x:str_preprocessing(x))

#Creating a new column which is a concatenation of Search Quuery, Product Title, Product Description
train['product_info'] = train['search_term']+"\t"+train['product_title'] +"\t"+train['product_description']

'''
Now we append different features to the training data, each one is described below:
1) Length of Query: Number of Words in the Search Query
2) Length of Title: Number of Words in the Product Title
3) Length of Description: Number of Words in the Product Description
4) Length of Brand: Number of Words in the Brand
5) Query in Title: Number of times the query occurs in the title
6) Query in Description: Number of times the query occurs in the description
7) Query last word in Title Count: Number of times the last word of the query occurs in the title
8) Query last word in Description Count: Number of times the last word of the query occurs in the description
9) Query last word in Title: '1' if the query last word occurs atleast once in the title, else zero
10) Query last word in Description: '1' if the query last word occurs atleast once in the description, else zero
11) Word in Title: 
12) Word in Description: 
13) Ratio Title: 
14) Ratio Description: 
15) Word in Brand:
16) Ratio Brand:
'''
train['len_of_query'] = train['search_term'].map(lambda x:len(x.split())).astype(np.int64)
train['len_of_title'] = train['product_title'].map(lambda x:len(x.split())).astype(np.int64)
train['len_of_description'] = train['product_description'].map(lambda x:len(x.split())).astype(np.int64)
train['len_of_brand'] = train['brand'].map(lambda x:len(x.split())).astype(np.int64)
train['query_in_title'] = train['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
train['query_in_description'] = train['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
train['query_last_word_in_title_count'] = train['product_info'].map(lambda x:str_common_word_count(x.split('\t')[0].split(" ")[-2],x.split('\t')[1]))
train['query_last_word_in_description_count'] = train['product_info'].map(lambda x:str_common_word_count(x.split('\t')[0].split(" ")[-2],x.split('\t')[2]))
train['query_last_word_in_title'] = train['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-2],x.split('\t')[1]))
train['query_last_word_in_description'] = train['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-2],x.split('\t')[2]))
train['word_in_title'] = train['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
train['word_in_description'] = train['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
train['ratio_title'] = train['word_in_title']/train['len_of_query']
train['ratio_description'] = train['word_in_description']/train['len_of_query']
train['attr'] = train['search_term']+"\t"+train['brand']
train['word_in_brand'] = train['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
train['ratio_brand'] = train['word_in_brand']/train['len_of_brand']

#Making the indices of the train uniform
train.index = range(train_size)

#Storing the labels in a variable
Y_labels = train['relevance'].values

#Splitting the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(train, Y_labels, test_size=0.3, random_state=42)
X_train.index = range(X_train.shape[0]) 
X_test.index = range(X_test.shape[0])
Y_train = pd.DataFrame(Y_train)
Y_test = pd.DataFrame(Y_test)

#Creating Document Term Matrices (DTM) for training data
count_vect_search = CountVectorizer()
train_dtm_search = count_vect_search.fit_transform(X_train['search_term'])
count_vect_title_description = CountVectorizer()
train_dtm_title_description = count_vect_title_description.fit_transform(X_train['product_title'] + X_train['product_description'])
count_vect_product_info = CountVectorizer()
train_dtm_product_info = count_vect_product_info.fit_transform(X_train['product_info'])

#Calculating the TF-IDF values from the DTM for training data
tf_transformer = TfidfTransformer(use_idf=True).fit(train_dtm_search)
train_tfidf_search = tf_transformer.transform(train_dtm_search)
#train_tfidf_search.shape = (30062, 5832)
tf_transformer = TfidfTransformer(use_idf=True).fit(train_dtm_title_description)
train_tfidf_title_description = tf_transformer.transform(train_dtm_title_description)
#train_tfidf_title_description.shape = (30062, 86136)
tf_transformer = TfidfTransformer(use_idf=True).fit(train_dtm_product_info)
train_tfidf_product_info = tf_transformer.transform(train_dtm_product_info)
#train_tfidf_product_info.shape = (30062, 88139)

#Performing dimensionality reduction using LSI and SVD on training data
svd_search = TruncatedSVD(n_components=50, random_state=42)
train_svd_search = svd_search.fit_transform(train_tfidf_search)
train_svd_search = Normalizer(copy=False).fit_transform(train_svd_search)
train_svd_search = pd.DataFrame(train_svd_search)
svd_title_description = TruncatedSVD(n_components=50, random_state=42)
train_svd_title_description = svd_title_description.fit_transform(train_tfidf_title_description)
train_svd_title_description = Normalizer(copy=False).fit_transform(train_svd_title_description)
train_svd_title_description = pd.DataFrame(train_svd_title_description)
svd_product_info = TruncatedSVD(n_components=50, random_state=42)
train_svd_product_info = svd_product_info.fit_transform(train_tfidf_product_info)
train_svd_product_info = Normalizer(copy=False).fit_transform(train_svd_product_info)
train_svd_product_info = pd.DataFrame(train_svd_product_info)

#Calculating Cosine Similarity between Search Query and Product Title & Description on training data
zero_data = np.zeros(shape=(train_svd_search.shape[0],1))
train_cosine_search_title_description = pd.DataFrame(zero_data)
for i in range(0,train_svd_search.shape[0]):
    train_cosine_search_title_description.ix[i,0] =  1 - spatial.distance.cosine(train_svd_search.ix[i,:], train_svd_title_description.ix[i,:])

#Creating Document Term Matrices (DTM) for testing data
test_dtm_search = count_vect_search.transform(X_test['search_term'])
test_dtm_title_description = count_vect_title_description.transform(X_test['product_title'] + X_test['product_description'])
test_dtm_product_info = count_vect_product_info.transform(X_test['product_info'])

#Calculating the TF-IDF values from the DTM for testing data
tf_transformer = TfidfTransformer(use_idf=True).fit(test_dtm_search)
test_tfidf_search = tf_transformer.transform(test_dtm_search)
#test_tfidf_search.shape = (12884, 5832)
tf_transformer = TfidfTransformer(use_idf=True).fit(test_dtm_title_description)
test_tfidf_title_description = tf_transformer.transform(test_dtm_title_description)
#test_tfidf_title_description.shape = (12884, 86136)
tf_transformer = TfidfTransformer(use_idf=True).fit(test_dtm_product_info)
test_tfidf_product_info = tf_transformer.transform(test_dtm_product_info)
#test_tfidf_product_info.shape = (12884, 88139)

#Performing dimensionality reduction using LSI and SVD on testing data
test_svd_search = svd_search.transform(test_tfidf_search)
test_svd_search = Normalizer(copy=False).fit_transform(test_svd_search)
test_svd_search = pd.DataFrame(test_svd_search)
test_svd_title_description = svd_title_description.transform(test_tfidf_title_description)
test_svd_title_description = Normalizer(copy=False).fit_transform(test_svd_title_description)
test_svd_title_description = pd.DataFrame(test_svd_title_description)
test_svd_product_info = svd_product_info.transform(test_tfidf_product_info)
test_svd_product_info = Normalizer(copy=False).fit_transform(test_svd_product_info)
test_svd_product_info = pd.DataFrame(test_svd_product_info)

#Calculating Cosine Similarity between Search Query and Product Title & Description on testing data
zero_data = np.zeros(shape=(test_svd_search.shape[0],1))
test_cosine_search_title_description = pd.DataFrame(zero_data)
for i in range(0,test_svd_search.shape[0]):
    test_cosine_search_title_description.ix[i,0] =  1 - spatial.distance.cosine(test_svd_search.ix[i,:], test_svd_title_description.ix[i,:])

#Deleting unwanted features from train data
train_important_features = X_train
train_important_features = train_important_features.ix[:,train_important_features.columns != 'id']
train_important_features = train_important_features.ix[:,train_important_features.columns != 'product_uid']
train_important_features = train_important_features.ix[:,train_important_features.columns != 'product_title']
train_important_features = train_important_features.ix[:,train_important_features.columns != 'search_term']
train_important_features = train_important_features.ix[:,train_important_features.columns != 'relevance']
train_important_features = train_important_features.ix[:,train_important_features.columns != 'product_description']
train_important_features = train_important_features.ix[:,train_important_features.columns != 'product_info']
train_important_features = train_important_features.ix[:,train_important_features.columns != 'brand']
train_important_features = train_important_features.ix[:,train_important_features.columns != 'attr']

#Deleting unwanted features from test data
test_important_features = X_test
test_important_features = test_important_features.ix[:,test_important_features.columns != 'id']
test_important_features = test_important_features.ix[:,test_important_features.columns != 'product_uid']
test_important_features = test_important_features.ix[:,test_important_features.columns != 'product_title']
test_important_features = test_important_features.ix[:,test_important_features.columns != 'search_term']
test_important_features = test_important_features.ix[:,test_important_features.columns != 'relevance']
test_important_features = test_important_features.ix[:,test_important_features.columns != 'product_description']
test_important_features = test_important_features.ix[:,test_important_features.columns != 'product_info']
test_important_features = test_important_features.ix[:,test_important_features.columns != 'brand']
test_important_features = test_important_features.ix[:,test_important_features.columns != 'attr']

#Creating various training data sets with appropriate features
pieces = [train_important_features,train_svd_search,train_svd_title_description]
X_train_1 = pd.concat(pieces,axis=1)
pieces = [train_important_features,train_svd_product_info]
X_train_2 = pd.concat(pieces,axis=1)
pieces = [train_important_features,train_svd_search,train_svd_title_description,train_cosine_search_title_description]
X_train_3 = pd.concat(pieces,axis=1)
pieces = [train_important_features,train_cosine_search_title_description]
X_train_4 = pd.concat(pieces,axis=1)
X_train_5 = train_important_features

#Creating various testing data sets with appropriate features
pieces = [test_important_features,test_svd_search,test_svd_title_description]
X_test_1 = pd.concat(pieces,axis=1)
pieces = [test_important_features,test_svd_product_info]
X_test_2 = pd.concat(pieces,axis=1)
pieces = [test_important_features,test_svd_search,test_svd_title_description,test_cosine_search_title_description]
X_test_3 = pd.concat(pieces,axis=1)
pieces = [test_important_features,test_cosine_search_title_description]
X_test_4 = pd.concat(pieces,axis=1)
X_test_5 = test_important_features

#Exporting the training data into CSV Files
X_train_1.to_csv('X_train_1.csv')
X_train_2.to_csv('X_train_2.csv')
X_train_3.to_csv('X_train_3.csv')
X_train_4.to_csv('X_train_4.csv')
X_train_5.to_csv('X_train_5.csv')
Y_train.to_csv('Y_train.csv')

#Exporting the testing data into CSV Files
X_test_1.to_csv('X_test_1.csv')
X_test_2.to_csv('X_test_2.csv')
X_test_3.to_csv('X_test_3.csv')
X_test_4.to_csv('X_test_4.csv')
X_test_5.to_csv('X_test_5.csv')
Y_test.to_csv('Y_test.csv')

#Training a Random Forest Model and predicitng on the test data
rfr = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 2)
model = rfr.fit(X_train_5,Y_train)
Y_predict = model.predict(X_test_5)

#Calculating the RMSE
RMSE = mean_squared_error(Y_test, Y_predict)**0.5

svr = SVR(kernel='linear', degree=3)
model_svr = svr.fit(X_train_1,Y_train.ix[:,0])


X_train_1 = pd.read_csv('X_train_1.csv')
X_train_2 = pd.read_csv('X_train_2.csv')
X_train_3 = pd.read_csv('X_train_3.csv')
X_train_4 = pd.read_csv('X_train_4.csv')
X_train_5 = pd.read_csv('X_train_5.csv')

X_test_1 = pd.read_csv('X_test_1.csv')
X_test_2 = pd.read_csv('X_test_2.csv')
X_test_3 = pd.read_csv('X_test_3.csv')
X_test_4 = pd.read_csv('X_test_4.csv')
X_test_5 = pd.read_csv('X_test_5.csv')


Y_train = pd.read_csv('Y_train.csv')
Y_test = pd.read_csv('Y_test.csv')

test = pd.read_csv('H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/Home Depot Dataset/test.csv')#[:1000] #update here
pro_desc = pd.read_csv('H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/Home Depot Dataset/product_descriptions.csv')#[:1000] #update here
attr = pd.read_csv('H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/Home Depot Dataset/attributes.csv')


train_dtm_search = train_dtm_search.todense()
dtm_svd_search = TruncatedSVD(n_components=50, random_state=42)
train_svd_search = dtm_svd_search.fit_transform(train_dtm_search)
train_svd_search = pd.DataFrame(train_svd_search)
train_svd_search.to_csv("train_svd_search.csv")

train_dtm_title_description = train_dtm_title_description.todense()
dtm_svd_title_description = TruncatedSVD(n_components=50, random_state=42)
train_svd_title_description = dtm_svd_title_description.fit_transform(train_dtm_title_description)
train_svd_title_description = pd.DataFrame(train_svd_title_description)
train_svd_title_description.to_csv("train_svd_title_description.csv")
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt

dataset = pandas.read_csv('empty.csv',nrows=2000)
df = DataFrame(dataset,columns=['Text','Tags'])
print(df)

#converting dataframe into array

data = df.values.tolist()
data = np.array(data)

def preprocess_data(data):
    print("Preprocessing data...")
    
    punc = string.punctuation           # Punctuation list
    sw = stopwords.words('english')     # Stopwords list
    for record in data:
        # Remove common punctuation and symbols
        for item in punc:
            record[0] = record[0].replace(item, "")
        # Lowercase all letters and remove stopwords 
        splittedWords = record[0].split()
        newText = ""
        for word in splittedWords:
            if word not in sw:
                word = word.lower()
                newText = newText + " " + word  
        record[0] = newText
        
    print("data preprocessing completed")        
    return data

def split_data(data):
    print("Splitting data...")
    
    features = data[:, 0]   # array containing all email text bodies
    labels = data[:, 1]     # array containing corresponding labels
    print(labels)
    training_data, test_data, training_labels, test_labels =\
        train_test_split(features, labels, test_size = 0.3, random_state = 2)
    
    print("Data splitted...!")
    return training_data, test_data, training_labels, test_labels

def get_count(text):
    wordCounts = dict()
    for word in text.split():
        if word in wordCounts:
            wordCounts[word] += 1
        else:
            wordCounts[word] = 1
    
    return wordCounts

def euclidean_difference(test_WordCounts, training_WordCounts):
    total = 0
    for word in test_WordCounts:
        if word in test_WordCounts and word in training_WordCounts:
            total += (test_WordCounts[word] - training_WordCounts[word])**2
            del training_WordCounts[word]
        else:
            total += test_WordCounts[word]**2
    for word in training_WordCounts:
            total += training_WordCounts[word]**2
    return total**0.5

def get_class(selected_Kvalues):
    dict1 = {}
    for value in selected_Kvalues:
        value=str(value[0])
        if value not in dict1:
            dict1[value]=0
        else:
            dict1[value] +=1
    Keymax = max(zip(dict1.values(), dict1.keys()))[1]
    return Keymax

def knn_classifier(training_data, training_labels, test_data, K, tsize):
    print("Running KNN Classifier...")
    
    result = []
    counter = 1
    # word counts for training Text
    training_WordCounts = [] 
    for training_text in training_data:
            training_WordCounts.append(get_count(training_text))
    for test_text in test_data:
        similarity = [] # List of euclidean distances
        test_WordCounts = get_count(test_text)  # word counts for test Text
        # Getting euclidean difference 
        for index in range(len(training_data)):
            euclidean_diff =\
                euclidean_difference(test_WordCounts, training_WordCounts[index])
            similarity.append([training_labels[index], euclidean_diff])
        # Sort list in ascending order based on euclidean difference
        similarity = sorted(similarity, key = lambda i:i[1])
        # Select K nearest neighbours
        selected_Kvalues = [] 
        for i in range(K):
            selected_Kvalues.append(similarity[i])
        # Predicting the class 
        result.append(get_class(selected_Kvalues))
    return result

def main(K,dataset):
    data = dataset
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)
    tsize = len(test_data)
    result = knn_classifier(training_data, training_labels, test_data[:tsize], K, tsize) 
    accuracy = (accuracy_score(test_labels[:tsize],result,normalize=False)/100)


    print("training data size\t: " + str(len(training_data)))
    print("labels" + str(test_labels))
    print("test data size\t\t: " + str(len(test_data)))
    print("K value\t\t\t\t: " + str(K))
    print("Samples tested\t\t: " + str(tsize))
    print("% Accuracy\t\t\t: " + str(accuracy*100))
    print("Number correct\t\t: " + str(int(accuracy * tsize)))
    print("Number wrong\t\t: " + str(int((1 - accuracy) * tsize)))
    return training_data, test_data, training_labels, test_labels,result,accuracy

def test_graph(acc,k):

    x = np.array(k)
    y = np.array(acc)

    colors_select=["red","yellow","green","gold","teal"]
    
    font1 = {'family':'serif','color':'gold','size':20}
    font2 = {'family':'serif','color':'slategray','size':15}
      
    plt.bar(x,y,width=0.4,color=colors_select)

    plt.title("Accuracy For Different K values",fontdict = font1)
    plt.xlabel("K_Values",fontdict = font2)
    plt.ylabel("ACCURACY SCORE",fontdict = font2)

    plt.show()

result_list1=[]
result_list2=[]
result_list3=[]
result_list4=[]

#comparing for different k values
for i in range(1,6):
   training_data, test_data, training_labels, test_labels,result,accuracy = main(i,data)
   result_list1.append(accuracy*100)
   result_list3.append(i)


for i in range(6,11):
   training_data, test_data, training_labels, test_labels,result,accuracy = main(i,data)
   result_list2.append(accuracy*100)
   result_list4.append(i)


test_graph(result_list1,result_list3)
test_graph(result_list2,result_list4)
from difflib import diff_bytes
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import csv
import re


def rem_html_tags(question):

    regex = re.compile('<.*?>')
    form= regex.sub('', question)
    return form

def removePunct(question):
    question = re.sub('\W+',' ', question)
    question = question.strip()
    return question

def tokenize(to_token):

  example_sent =to_token
 
  stop_words = set(stopwords.words('english'))
 
  word_tokens = word_tokenize(example_sent)
 
  filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
 
  filtered_sentence = []
 
  for w in word_tokens:
   if w not in stop_words:
     filtered_sentence.append(w)
 
  return filtered_sentence

def listToString(s): 
    str1 = " " 
    return (str1.join(s))

global df
df = pd.read_csv('stackoverflow_new.csv')
def preprocess(df):
  
    size = 0
    
    csv_file =  open("empty.csv",'w',newline='')

    text_list=[]
    
    for i in range(len(df)):
        
        size = size + 1
        question = df['Text'][i]


        tokens=tokenize(question)
        text_pre =listToString(tokens)
        #print(text_pre)
        temp=[]
        temp.append(text_pre)
        text_list.append(temp)
    with csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(text_list)

    print("preprocess completed")

preprocess(df)
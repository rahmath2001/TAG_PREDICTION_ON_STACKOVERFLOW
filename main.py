from asyncio.windows_events import NULL
from tkinter import messagebox     #All imports
from tkinter import *
import tkinter
import matplotlib.pyplot as plt
from PIL import ImageTk,Image
import pandas as pd 
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score,mean_squared_error
import math

#read dataset

df = pd.read_csv('empty.csv')
df.head()

df['Tags']=df['Tags'].apply(lambda x: ast.literal_eval(x))
y= df['Tags']
multilabel= MultiLabelBinarizer()         #multiLabelBinarier for Tags
y = multilabel.fit_transform(df['Tags'])
print(y)
print(multilabel.classes_)
pd.DataFrame(y,columns=multilabel.classes_)

#vectorization

tfidf= TfidfVectorizer(analyzer='word',max_features=10000)
X=tfidf.fit_transform(df['Text'])
print(X)
print(X.shape, y.shape)
print(" ")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.35,random_state=10)

#declaration of classifiers


sgd = SGDClassifier()
lr = LogisticRegression()
svc=LinearSVC()
knn=KNeighborsClassifier()

acc_text=[]

#matplotlib functions
def accuracy_graph(acc):

    def acc_val(X_axis,Y_axis):
        for a in range(len(X_axis)):
            plt.text(a,Y_axis[a],Y_axis[a],ha="center")

    x = np.array([ "SGD", "LOGISTIC REGRESSION","SVM","KNN"])
    y = np.array(acc)


    font1 = {'family':'serif','color':'gold','size':20}
    font2 = {'family':'serif','color':'slategray','size':15}
     
    plt.bar(x,y,width=0.4,color="teal")

    acc_val(x,y)

    plt.title("ACCURACY COMPARISON GRAPH",fontdict = font1)
    plt.xlabel("CLASSIFIERS",fontdict = font2)
    plt.ylabel("ACCURACY SCORE",fontdict = font2)
    #plt.text(x+width/2,y+height*1.01,height,ha='center',weight='bold')

    plt.show()


def j_score(y_true,y_pred):
    jaccard = np.minimum(y_true,y_pred).sum(axis = 1)/np.maximum(y_true,y_pred).sum(axis=1)
    return jaccard.mean()*100

def print_score(y_pred,clf):
    print("Classifier: ",clf.__class__.__name__)
    print('Jaccard score: {}'.format(j_score(y_test,y_pred)))
    precision= precision_score(y_test,y_pred,average='macro') * 100
    print('Precision: %f' % precision)
    recall= recall_score(y_test,y_pred,average='macro')  * 100
    print('Recall: %f' % recall)
    f1= f1_score(y_test,y_pred,average='macro') * 100
    print('F1 Score: %f' % f1)

def accuracy_score_c(accuracy_c):
    #print("Classifier: ",clf._class.name_)
    print('accuracy score: {}'.format(accuracy_c))
    print("  ")
    print(" ")

def mean_error(error_rate,clf):
    if (clf != lr):
        print('Root mean squared error: {}'.format(error_rate))


acc_list=[]

for classifier in [sgd,lr,svc,knn]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print_score(y_pred,classifier)
    acc_score = (accuracy_score(y_test, y_pred,normalize=False)/100)

    m_a_e = math.sqrt((mean_squared_error(y_test,y_pred)*100)) #rmse calculation
    mean_error(m_a_e,classifier)
    accuracy_score_c(acc_score)
    acc_list.append(acc_score)

#predicting the model with samples

def suggestion(abc):
    x=[]
    x.append(abc)
    xt=tfidf.transform(x)
    clf.predict(xt)
    answ = multilabel.inverse_transform(clf.predict(xt))

    for i in range(0,len(answ)+2):
      for j in answ:  
        if(i==0):  

         print(j[i])
         tag_Label1.configure(text = j[i])
         i+1
         continue
        elif(i==1):
         tag_Label2.configure(text=j[i])
         i+1
         continue
        else:
         tag_Label3.configure(text=j[i])
         
#Gui work

root=Tk()
root.geometry("1366x768+0+0")
root.resizable(True,True)
root.state("zoomed")
root.title("Tag Prediction on StackOverflow")
root.config(bg="white")

icon=PhotoImage(file = 'stack.png')
root.iconphoto(False,icon)

imge = PhotoImage(file =r"stack.png")     

mainFrame=Frame(root,bg="white")
mainFrame.place(x=200,y=50,width="966",height="600")

#Background image fit
img=ImageTk.PhotoImage(Image.open("so2.jpeg")) #bg image
label_img = Label(mainFrame,image=img)
label_img.pack()

#title image
image_label =Label(image=imge).place(x=210,y=60,height=70,width=90)

titleLabel=Label(mainFrame,bg="white",fg="#000080",text="Tag Prediction On Stack Overflow",font=("lato",20,"bold"))
titleLabel.place(x=10,y=10,width="946",height="70")

urlLabel=Label(mainFrame,text="Enter your question   :",font=("tahoma",15))
urlLabel.place(x=10,y=140)

urlText=Text(mainFrame,bg="white",fg="#006666")
urlText.place(x=250,y=105,width="600",height="100")
urlText.configure(font=("courier",15,"italic"))

tagLabel=Label(mainFrame,fg="black",text="Suggested Tags :  ",font=("tahoma",15))
tagLabel.place(x=10,y=360)

#tag_Label1,tag_Label2,tag_Label3 to showcase the predicted tags

tag_Label1=Label(mainFrame,fg="black",bg="yellow",text="",font=("courier",14))
tag_Label1.place(x=220,y=360)

tag_Label2=Label(mainFrame,fg="black",bg="yellow",text="",font=("courier",15))
tag_Label2.place(x=320,y=360)

tag_Label3=Label(mainFrame,fg="black",bg="yellow",text="",font=("courier",15))
tag_Label3.place(x=425,y=360)

#function to accuracy_graph 

def get_graph():
    accuracy_graph(acc_list)

generate_graph_button = Button(mainFrame,text = 'Generate Graph',bd ='5',bg='orange',pady=5,command=get_graph)
generate_graph_button.pack(side='top')
generate_graph_button.place(x=220,y=500)


def suggest():
    x =urlText.get(1.0,"end-1c")
    if(not x):
        messagebox.showinfo("showinfo", "Enter question")
    else:
        suggestion(x)


Suggest_tagbutton= Button(mainFrame,text = 'Suggest tags',bd ='5',bg='orange',pady=5,command=suggest)
Suggest_tagbutton.pack(side='top')
Suggest_tagbutton.place(x=400,y=500)

accuracy_button= Button(mainFrame,text ='Get Accuracy',bd ='5',bg='orange',pady=5)
accuracy_button.pack(side='top')
accuracy_button.place(x=580,y=500)

root.mainloop()
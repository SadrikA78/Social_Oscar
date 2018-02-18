# Social_Oscar
 coding: utf8
from Tkinter import *
import Tkinter as tk
from PIL import ImageTk, Image
import os
#from Tkinter import Button
import webbrowser
import twitter
import json
import time
import nltk
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import numpy as np
import pandas as pd
import re
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models, similarities 
import gensim
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from collections import Counter
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
#---------------------------------------------------------------обучение определения тональности
def show_entry_fields():
    a=1911650800   
    if int(time.time())<a:
        callback()
    else:
        print u'конец демо'
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent.decode('utf-8'))})
pos = []
with open("pos_tweets.txt") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])
neg = []
with open("neg_tweets.txt") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])
training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]
classifier = NaiveBayesClassifier.train(training)
classifier.show_most_informative_features()
def callback():
    #----------------------------------------------------------------------обращение к API Twitter
    print u'Отправляется запрос к API: ', e1.get()
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    OAUTH_TOKEN = ''
    OAUTH_TOKEN_SECRET = ''
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)
    twitter_api = twitter.Twitter(auth=auth)
    tweet=twitter_api.search.tweets(q=(e1.get()), count="100")
    p = json.dumps(tweet)
    res2 = json.loads(p)
    print u'Получено сообщений: ',len(res2['statuses'])
    #----------------------------------------------------------------------формирование окна выдачи результатов
    stopwords = nltk.corpus.stopwords.words('english')
    en_stop = get_stop_words('en')
    stemmer = SnowballStemmer("english")
    #print stopwords[:10]
    total_word=[]
    lang=[]
    i=0
    retweet_count=[]
    followers_count=[]
    friends_count=[]
    tonal=[]

    
    window = tk.Toplevel(root)
    window.minsize(1300,1000)
    window.title(u"Вывод данных")
    #webbrowser.open("index.html")
    w00=Label(window,text=u"СООБЩЕНИЯ", font = "Times")
    w00.place(relx=0.2, rely=0.01)
    t1=Text(window, height=30, width=75)
    t1.place(relx=0.01, rely=0.03)
    while i<len(res2['statuses']):
        tweet=str(i+1)+') '+str(res2['statuses'][i]['created_at'])+' '+(res2['statuses'][i]['text'])+'\n'
        t1.insert(END, (tweet))
        lang.append(res2['statuses'][i]['lang'])
        retweet_count.append(res2['statuses'][i]['retweet_count'])
        followers_count.append(res2['statuses'][i]['user']['followers_count'])
        friends_count.append(res2['statuses'][i]['user']['friends_count'])
        tokenizer = RegexpTokenizer(r'\w+')
        if 'en' in res2['statuses'][i]['lang']:
           tonal.append(classifier.classify(format_sentence(res2['statuses'][i]['text'].encode('utf-8'))))
        total_word.extend(tokenizer.tokenize(res2['statuses'][i]['text']))        
        i=i+1
    #--------------------------------------------------------------------------оцека распространения
    print u'Количество ретвитов', sum(retweet_count)
    print u'Возможный охват', sum(followers_count)+sum(friends_count)
    w0=Label(window,text=u"РАСПРОСТРАНЕНИЕ", font = "Times")
    w0.place(relx=0.55, rely=0.01)
    w2=Label(window,text=u"Количество заимствований:", font = "Times")
    w2.place(relx=0.5, rely=0.03)
    t2=Text(window, height=1, width=7)
    t2.place(relx=0.67, rely=0.035)
    t2.insert(END,sum(retweet_count))
    w3=Label(window,text=u"Возможный охват:", font = "Times")
    w3.place(relx=0.5, rely=0.05)
    t3=Text(window, height=1, width=7)
    t3.place(relx=0.67, rely=0.055)
    t3.insert(END,(sum(followers_count)+sum(friends_count)))

    #--------------------------------------------------------------------------анализ языка публикаций
    w7=Label(window,text=u"АНАЛИЗ ЯЗЫКА ПУБЛИКАЦИЙ", font = "Times")
    w7.place(relx=0.65, rely=0.1)
    f = Figure(figsize=(6, 4))
    a = f.add_subplot(111)
    t = Counter(lang).keys()
    y_pos = np.arange(len(t))
    performance = Counter(lang).values()
    error = np.random.rand(len(t))
    s = Counter(lang).values()
    a.barh(y_pos,s)
    a.set_yticks(y_pos)
    a.set_yticklabels(t)
    a.invert_yaxis()
    a.set_ylabel(u'Язык сообщений')
    a.set_xlabel(u'Количество')
    canvas = FigureCanvasTkAgg(f, master=window)
    canvas.show()
    canvas.get_tk_widget().place(relx=0.52, rely=0.12)#pack(side=TOP, fill=BOTH, expand=1)
    canvas._tkcanvas.place(relx=0.52, rely=0.12)#pack(side=TOP, fill=BOTH, expand=1)
 
    #--------------------------------------------------------------------------оцека тональности
    tonal_pos= Counter(tonal).keys()
    tonal_value=Counter(tonal).values()
    print u'Распределение тональности', tonal_pos, tonal_value
    w4=Label(window,text=u"РАСПРЕДЕЛЕНИЕ ТОНАЛЬНОСТИ", font = "Times")
    w4.place(relx=0.75, rely=0.01)
    w5=Label(window,text=u"Количество негатитва:", font = "Times")
    w5.place(relx=0.75, rely=0.03)
    t5=Text(window, height=1, width=7)
    t5.place(relx=0.9, rely=0.035)
    print tonal_value
    if tonal_value[0]>0:
        t5.insert(END,tonal_value[0])
    else:
        print u'тональность выявляет только на англиском'
    w6=Label(window,text=u"Количество позитива:", font = "Times")
    w6.place(relx=0.75, rely=0.05)
    t6=Text(window, height=1, width=7)
    t6.place(relx=0.9, rely=0.055)
    if tonal_value[1]>0:
        t6.insert(END,tonal_value[1])
    else:
        print u'тональность выявляет только на англиском'
root = Tk()
root.minsize(900,580)
img = ImageTk.PhotoImage(Image.open("main_window_screen.png"))
panel = Label(root, image = img)
w1=Label(root, text=u"Введите запрос", font = "Times 16 bold")
w1.place(relx=0.45, rely=0.6, anchor=SE)
e1 = Entry(root)
e1.place(relx=0.6, rely=0.6, anchor=SE)
panel.pack(side = "bottom", fill = "both", expand = "yes")
b = Button(root, text=u"Перейти к поиску", font = "Times 14 bold", command=show_entry_fields)
b.place(relx=0.9, rely=0.91, anchor=SE)
#root.geometry("500x500")
#app = App(root)   
root.mainloop()

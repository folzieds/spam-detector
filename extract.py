# Extracting mail subject and body
#Updated from Extracted content.py

import email, email.parser
import pandas as pd
import re, nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

word_lemmitizer = WordNetLemmatizer()
Tf = TfidfVectorizer(decode_error= 'ignore')
stop_words = stopwords.words('english')

with open(r'spam-mail.tr.label','r') as file_2:
    label = email.message_from_file(file_2)
    
label_list =str(label.get_payload()).split()

label_sort = []
target_list = []

for word in label_list:
    label_sort.append(word.split(','))
for word in label_sort:
    target_list.append(word[1])

# Ham is 1 and spam is 0   
target_df = pd.Series(target_list[1:]) 

del target_list, word, label_list, label_sort 

def ExtractSub(filename):
    ''' 
        Extracts the subject from the .eml file
    '''
    msg = email.message_from_file(filename)
    subject = str(msg.get('subject'))
    body = msg.get_payload()
    
    if type(body) == type(list()):
        body = body[0]
    if type(body) != type(str()):
        body = str(body)
    mail = f'{subject} {body}'
    return mail

number = 1
dataTr = []

while number <= 2500:
    try:
        with open(f"TR-mails\\TR\\TRAIN_{number}.eml","r") as file:
            dataTr.append(ExtractSub(file))
    except UnicodeDecodeError:
        try:
            with open(f"TR-mails\\TR\\TRAIN_{number}.eml","r", encoding="utf-8") as file:
                dataTr.append(ExtractSub(file))
        except UnicodeDecodeError:
            with open(f"TR-mails\\TR\\TRAIN_{number}.eml","r", encoding="latin-1") as file:
                dataTr.append(ExtractSub(file))
    number += 1

num = 1
dataTs = []

while num <= 1827:
    try:
        with open(f"TT-mails\\TT\\TEST_{num}.eml","r") as file:
            dataTs.append(ExtractSub(file))
    except UnicodeDecodeError:
        try:
            with open(f"TT-mails\\TT\\TEST_{num}.eml","r", encoding="utf-8") as file:
                dataTs.append(ExtractSub(file))
        except UnicodeDecodeError:
            with open(f"TT-mails\\TT\\TEST_{num}.eml","r", encoding="latin-1") as file:
                dataTs.append(ExtractSub(file))
    num += 1

del num, number

def RemoveHTMLTags(data):
    '''Remove HTML Tags from the file using Regex requires the libary re'''
    
    s = re.sub(r"[^a-zA-z']", ' ', data)
    return s


def my_tokenizer(s):
    s = RemoveHTMLTags(s)
    s = s.lower()
    token = nltk.tokenize.word_tokenize(s)
    token = [t for t in token if len(t) > 2]
    token = [word_lemmitizer.lemmatize(t) for t in token]
    token = [t for t in token if t not in stop_words]
    token = [t for t in token if not any(c.isdigit() for c in t)]
    return ' '.join(token)

def convertToToken(data):
    data_x = []
    for words in data:
        tokenize = my_tokenizer(words)
        data_x.append(tokenize)
    return data_x



dataTr_x = []
dataTs_x = []

dataTr_x = convertToToken(dataTr)

dataTs_x = convertToToken(dataTs)

x_train = Tf.fit_transform(dataTr_x)
x_test = Tf.transform(dataTs_x)

from sklearn.naive_bayes import MultinomialNB
Nb = MultinomialNB()
Nb.fit(x_train, target_df)

y_pred = Nb.predict(x_test)

print(f'model score: {Nb.score(x_test,y_pred)}')


# kaggle submission
Submit = pd.DataFrame(y_pred, columns = ['Prediction'])
Submit.index += 1 

Submit.to_csv('output.csv')


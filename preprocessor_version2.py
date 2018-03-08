'''
prerequisites:
    - The datafiles must be in the same folder as the script. This can be interactive later with tkinter (though this tends to be defective on linux machines)
    - Already specified in def main(): use train.csv and test_with_solutions.csv
    - Output may look weird in the dicts, but that is because the stem of words are used. These stems will reduce the redundancy of the data (e.g.: 'fuck', 'FUCK', 'fucking' etc. becomes 'fuck')

TODO:
    - implement algorithm
'''
import csv
from nltk.stem.porter import *
import nltk
import re
import codecs
import unicodedata
import pandas as pd
import math
#for the first run, uncomment the line below
#nltk.download('stopwords')
stemmer = PorterStemmer()

def readfile(file):
    #put the csv file in a dict with key = ID, value = [insult,comment]
    Comments = {}
    setOfWords = set()
    with open(file,'r') as f:
        csvfile = csv.DictReader(f,delimiter=",",quotechar='\"')
        ID = 1
        for row in csvfile:
            insult = row['Insult']
            comment = row['Comment']
            preprocomment = preprocess(comment)
            setOfWords.update(preprocomment)
            IDkey = str(ID)
            Comments[IDkey] = [insult,preprocomment]
            ID += 1
    print(len(setOfWords))
    f.close()
    return Comments,setOfWords    
            
def preprocess(comment):
    #preprocess words: tokenize and stem according to english
    comment = comment.replace('\\\\','\\')
    newcomment = unicodedata.normalize('NFKD', comment).encode('ascii', 'ignore')
    newcomment = newcomment.decode('unicode-escape','ignore')
    newcomment = newcomment.encode('ascii','ignore')
    newcomment = newcomment.decode('ascii','ignore')
    words = " ".join(newcomment.split())
    words = words.split(" ")
    cleanedcomment = []
    for word in words:
        cleanedword = str(re.sub('\W+','', word))
        stemmedword = stemmer.stem(cleanedword).lower()
        if stemmedword != '':
            cleanedcomment.append(stemmedword)
    return cleanedcomment
         
def make_tfcounts(comments, uniquewords):
    #uniquewords = list(uniquewords)
    print("Making TF")
    tfdict = dict()
    #count the number of times 'word' occurs in a single comment    
    for word in uniquewords:
        iddict = dict()
        for key, value in comments.items():
            counter = 0
            for w in value[1]:
                if w == word:
                    counter += 1
            iddict[key] = counter
        tfdict[word] = iddict
    return tfdict

def make_dfcounts(comments, uniquewords):
    #uniquewords = list(uniquewords)
    print("Making DF")
    dfdict = dict()
    
    #count the number of comments 'word' appears in.
    for word in uniquewords:
        counter = 0
        for key, value in comments.items():
            if word in value[1]:
                counter +=1    
        dfdict[word] = counter
    return dfdict

def make_tfidf(tfdict,dfdict):
    print("Making TF-IDF")
    tfidfdict = dict()
    #get total number of comments
    numberOfComments = float(len(tfdict['a']))
    #construct tfidf
    for word,IDs in tfdict.items():
        commentID = dict()
        for i,value in IDs.items():
            tf = tfdict[word][i]
            df = dfdict[word]
            tfidf = float(tf) * math.log10(numberOfComments/float(df))
            commentID[i] = tfidf
        tfidfdict[word] = commentID
    return tfidfdict    

#BEGIN: THIS FUNCTION IS ONLY FOR TESTING
#def get_bestword(tfidfdict):
#    print("Getting best word")
#    topvalue = 0
#    topword = ''
#    topcomment = ''
#    #currently, writes the tfidf as an extremely large csv file and prints the highest scoring tfidf entry
#    with open("all_tfidf_values.csv",'w') as w:
#        for word,ID in tfidfdict.items():
#            for i,value in ID.items():
#                w.write(",".join([word,i,str(value)])+"\n")
#                if value > topvalue:
#                    topvalue = value
#                    topcomment = i
#                    topword = word
#    w.close()
#    print(topword,topcomment,topvalue)
##END

def make_dataframe_and_save(tfidfdict):
    #make a dataframe from the tfidf and save it to a csv file
    print("Making dataframe")
    dataframeTFIDF = pd.DataFrame.from_dict(tfidfdict)
    dataframeTFIDF.to_csv("tfidf_dataframe.csv")

def main():
    trainingdatafile = 'train.csv'
    testdatafile = 'test_with_solutions.csv'
    trainingDict, setOfWords = readfile(trainingdatafile)
    tfdict = make_tfcounts(trainingDict,setOfWords)
    dfdict = make_dfcounts(trainingDict,setOfWords)
    tfidfdict = make_tfidf(tfdict,dfdict)
    make_dataframe_and_save(tfidfdict)
    #get_bestword(tfidfdict)
    testDict = readfile(testdatafile)
    
main()

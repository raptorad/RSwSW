# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:33:21 2017

@author: Michal
"""
path='G:\WinPython-64bit-3.4.3.5\RSwSW\SAWSDL-TC3_WSDL11'
from lxml import etree as ET
from rdflib import Graph
import numpy as np
import pprint
"""
for offers in root.iter('offers'):
    for offer in offers.iter('offer'):
        print(offer.tag)
"""


def FindComplexTypes(uri):
    wtree=ET.parse(path+"\htdocs\services\sawsdl_wsdl11"+'\\' +uri)
    print(path+"\htdocs\services\sawsdl_wsdl11"+'\\' +uri)
    r=wtree.getroot()
    ns=r.nsmap
    del ns[None]
    ct=r.findall(".//xsd:complexType",ns)
    return ct

def printXML(elem):
    print(ET.tostring(elem, pretty_print=False))

def GetOntologyURI(complexType):     
    atr=complexType.attrib
    return atr['{'+complexType.nsmap["sawsdl"]+'}modelReference']

def OntologyFromURI(uri):
    splt=uri.split('/')
    last=splt[-1].split('#')
    fileName=last[0]
    hsh='#'+last[-1]
    print ("fileName:",fileName," hsh:",hsh)
    ontoPath=path+"\htdocs\ontology"
    g = Graph()
    g.parse(ontoPath+"\\"+fileName, format="xml")
    for stmt in g:
        print(stmt)
        #pprint.pprint(stmt)
def OntologyFromURIXML(uri):
    splt=uri.split('/')
    last=splt[-1].split('#')
    fileName=last[0]
    hsh='#'+last[-1]
    ontoPath=path+"\htdocs\ontology"
    comments=[]    
    try:    
        t=ET.parse(ontoPath+"\\"+fileName)
        r=t.getroot()
        ns=r.nsmap
        del ns[None]
        comments=r.findall(".//rdfs:comment",ns)
    except OSError as err:
        print("OS error: {0}".format(err))
    return comments

def URItexts():
    tree = ET.parse(path+'\sawsdl-tc3.xml')
    root = tree.getroot()
    uris=root.findall("./offers/offer/uri")
    uritexts=[]
    for uri in uris:
        uritexts.append(uri.text.split('/')[-1])
    return uritexts
#def ListAllComplexTypes():

def LoadRequests():
    tree = ET.parse(path+'\sawsdl-tc3.xml')
    root = tree.getroot()
    requests=root.findall('./relevancegradeexport/binaryrelevanceset/request')
    return requests

#zwraca listę dwoch list pierwsza to lista offers ID, a druga relewancja
def OffersRatingsFromRequest(request):
    of=request.findall('./ratings/offer')
    l=[]
    ofs=[]
    r=[]
    for o in of:          
        offerID=int(o.attrib['id'])
        ofs.append(offerID)                
        rel=int(o.findall('./relevant')[0].text)
        r.append(rel)
    l.append(ofs)
    l.append(r)
    return l

def OffersAndRequestURIS():
    ofUris=URItexts()
    ret=[]
    reqs=LoadRequests()
    for req in reqs:
        element=[]
        of=req.findall('./ratings/offer')
        print(len(of))
        for o in of:          
            offerID=int(o.attrib['id'])-1                
            rel=int(o.findall('./relevant')[0].text)
            if offerID<len(ofUris):
                ofText=ofUris[offerID]
                uri=req.findall('./uri')[0].text
                element.append(ofText)
                element.append(uri)
                element.append(rel)
                ret.append(element)
    return ret
def OffersAndRequestURISToFile(outPath):
    data=OffersAndRequestURIS()   
    F = open(outPath,'w')
    F.write("\"requests\":[\n")
    for row in data:
        F.write("{\"offerURI\":\""+row[0]+"\",\"requestURI\":\""+row[1]+"\",\"relvance\":"+str(row[2])+"},\n")
    F.write("]")   
#tworzy tablicę numpy z requests na offers z binarną oceną relewancji
def CreateTable(requests,offersLen):
    tab=np.zeros((len(requests),offersLen))
    for r  in requests:
        rid=int(r.attrib['id'])
        orfr=OffersRatingsFromRequest(r)
        ofids=orfr[0]
        rel=orfr[1]
        i=0        
        for o in ofids:
            if o < 1080 :
                tab[rid-1,o-1]=rel[i]
            i=i+1
    return tab
            
    
"""    
uritexts=URItexts()
allComplexTypes=FindComplexTypes(uritexts[0])
for i in range(1,100):
    allComplexTypes+=(FindComplexTypes(uritexts[i]))
allComments=OntologyFromURIXML(GetOntologyURI(allComplexTypes[0]))
for i in range(1,len(allComplexTypes)):
    allComments+=OntologyFromURIXML(GetOntologyURI(allComplexTypes[i]))
texts=[]
for c in allComments:
    texts.append(c.text)       
cts=FindComplexTypes(uritexts[3])
com=OntologyFromURIXML(GetOntologyURI(cts[0]))
"""
"""
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(texts)
from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
from sklearn.naive_bayes import MultinomialNB

target=np.array(np.zeros(len(texts)))
clf = MultinomialNB().fit(X_train_tfidf, target)
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf = text_clf.fit(texts, target)
"""
"""
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
_ = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print (np.mean(predicted == twenty_test.target)) 
"""

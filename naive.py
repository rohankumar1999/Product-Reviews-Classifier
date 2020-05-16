import string
import random
from math import log
import numpy as np
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 
k_folds=5

def count(word,train_set,t):		#function to calculate the word count.
	c=0
	for review in train_set:
		if (word in review) and (review[-1]==t):
			c+=1
	return c
def classify(train_set,test_set):
	y_1=0
	y_0=0
	tp=fp=fn=tn=0
	for review in train_set:
		
		if(review[-1]=='1'):
			y_1=y_1+1
		if(review[-1]=='0'):
			y_0=y_0+1
	
	for review in test_set:
		p0=log(y_0/(y_0+y_1))
		p1=log(y_1/(y_1+y_0))
		for i in range(len(review)-1):
			
				p0+=log((count(review[i],train_set,'0')+1)/(y_0+3543))  #computing the product of probabilities of the word if it belongs to 0 class 
			
				p1+=log((count(review[i],train_set,'1')+1)/(y_1+3543))	#computing the product of probabilities of the word if it belongs to 1 class
		
		pred='1' if p1>p0 else '0'		#classifying the review in test set based on  the probability of it lying in  either class.

		# print(review,pred)
		if pred==review[-1] and pred=='1':
			tp+=1
		elif pred=='0' and pred!=review[-1]:
			fn+=1
			
		elif pred=='1' and review[-1]=='0':
			fp+=1
		elif pred=='0' and review[-1]=='0':
			tn+=1
		
	precision=(tp/(tp+fn))
	recall=(tp/(tp+fp))
	print(precision,recall)
	f1_score=(2*precision*recall)/(precision+recall) # computing the accuracy measures
	print('accuracy: ',(tp+tn)/(tp+tn+fp+fn))
	return f1_score,(tp+tn)/(tp+tn+fp+fn)
def evaluate(ip):
	f1_score=list()
	accuracy=list()
	for fold in ip:
		train_set = list(ip)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			# row_copy[-1] = None
		
		f1,acc=classify(train_set,test_set)
		f1_score.append(f1)
		accuracy.append(acc)
	f1_score=np.array(f1_score)
	accuracy=np.array(accuracy)
	print('f1_score: ',np.mean(f1_score),'+/-',np.std(f1_score))
	print('accuracy: ',np.mean(accuracy),'+/-',np.std(accuracy))
def cross_validation_split(dataset, n_folds):
    """Split dataset into the k folds. Returns the list of k folds"""
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
												# reading the input from text file.
f=open('a1_d3.txt','rt')
fr=f.readlines()
f.close()
ip=list()
for line in fr:
	line=line.replace('.',' ')
	for char in string.punctuation:
		line=line.replace(char,'')
	ip.append(line.strip('\n').split()) #splitting each review into its component words ,you can extend this by splitting into n-grams.

ip=cross_validation_split(ip,k_folds)

# function call to classify the reviews.
evaluate(ip)
# print(correct,total)

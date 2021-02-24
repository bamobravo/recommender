"""
This file will contain a collection of function for manipulating the data
"""

import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
training_data_path='saved/training.csv'
testing_data_path = 'saved/testing.csv'
import pickle

lookup ={}
filepath='saved/lookup.pickle'
if os.path.isfile(filepath):
	with open(filepath,'rb') as fl:
		lookup = pickle.load(fl)

def getIndex(value, array,columnType=False):
	try:
		if columnType=='genre':
			vals = value.split(',')
			temp= [str(array.index(x)+1) for x in vals]
			res=','.join(temp)
			return res
		result = array.index(value)
		return str(result+1)
	except Exception as e:
		return '-1'
	
def transform_function(row):
	row['gender'] = getIndex(row['gender'],lookup['gender'])
	row['occupation']= getIndex(row['occupation'],lookup['occupation'])
	row['zip_code'] = getIndex(row['zip_code'],lookup['zip_code'])
	row['age_class'] = getIndex(row['age_class'],lookup['age_class'])
	row['genre'] = getIndex(row['genre'],lookup['genre'],columnType='genre')
	row['CompanionContext'] = getIndex(row['CompanionContext'],lookup['CompanionContext'])
	return row

def loadList(path):
	directory='datasets/ml-100k/'
	fullpath = directory+path
	result=[]
	with open(fullpath,'r') as fl:
		text = fl.read()
		result= [x.strip() for x in text.split('\n')]
		return result

def hasGenre(val,pat):
	values = val.split(',')
	return pat in values
	
def buildLookup():
	heading=['zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	result={}
	result['occupation']=loadList('u.occupation')
	result['']
	print(result['occupation'])
	exit()
	lookup = {x:list(all_data[x].unique()) for x in heading}
	temp_genre =[]
	lookup['genre']=['comedy','short', 'sport','history', 'biography', 'family', 'music', 'unknown', 'action','adventure', 'animation', 'children',  'crime', 'documentary','drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery','romance', 'sci-fi', 'thriller', 'war', 'western']
	with open(filepath,'wb') as fl:
		pickle.dump(lookup,fl)

def transformNumeric(all_data):
	lookup ={}
	filepath='saved/lookup.pickle'
	if os.path.isfile(filepath):
		with open(filepath,'rb') as fl:
			lookup = pickle.load(fl)
	else:
		heading=['zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
		lookup = {x:list(all_data[x].unique()) for x in heading}
		temp_genre =[]
		lookup['genre']=['comedy','short', 'sport','history', 'biography', 'family', 'music', 'unknown', 'action','adventure', 'animation', 'children',  'crime', 'documentary','drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery','romance', 'sci-fi', 'thriller', 'war', 'western']
		with open(filepath,'wb') as fl:
			pickle.dump(lookup,fl)
	# perform conversion for genre movie
	print('need to process ',all_data.shape[0])
	for index, row in all_data.iterrows():
		all_data.at[index,['gender']] = getIndex(row['gender'],lookup['gender'])
		all_data.at[index,['occupation']]= getIndex(row['occupation'],lookup['occupation'])
		all_data.at[index,['zip_code']] = getIndex(row['zip_code'],lookup['zip_code'])
		all_data.at[index,['age_class']] = getIndex(row['age_class'],lookup['age_class'])
		all_data.at[index,['genre']] = getIndex(row['genre'],lookup['genre'],columnType='genre')
		all_data.at[index,['CompanionContext']] = getIndex(row['CompanionContext'],lookup['CompanionContext'])
		print('processing ',index+1)
	return all_data

def combine_data(filepath):
	directory='datasets/ml-100k/'
	u_data = pd.read_csv(directory+filepath,sep='\t',names=['user_id','movie_id','rating','rating_time'])
	movie_genres=['unknown','action','adventure','animation','children','comedy','crime','documentary','drama','fantasy','film-noir','horror','musical','mystery','romance','sci-fi','thriller','war','western']
	movie_info=['movie_id','movie_title','release_date','video_release_date','IMDB_url']
	movie_header= movie_info+movie_genres
	u_movies=pd.read_csv(directory+"u.item",sep='|',names=movie_header)
	u_movies=combine_genre(u_movies)
	user_heading = ['user_id','age','gender','occupation','zip_code']
	users_info = pd.read_csv(directory+"u.user",sep='|',names=user_heading)

	# now lets join the data together
	rating_movies = pd.merge(u_data,u_movies,on='movie_id')
	all_data = pd.merge(rating_movies,users_info,on='user_id')
	all_data['age_class']=all_data['age'].astype('int').floordiv(10,fill_value='1')
	all_data['zip_code']=all_data['zip_code'].astype('int').floordiv(100,fill_value=1000)
	all_data['rated']=all_data['rating']>3
	all_data = all_data.astype({'rated':'int'})
	companions = list(all_data['CompanionContext'].unique())
	all_occupation = load_occupation_category()
	all_data['CompanionContext']=all_data['CompanionContext'].apply(lambda x,arr: arr.index(x),args=(companions,))
	all_data['occupation']=all_data['occupation'].apply(lambda x,arr: arr.index(x),args=(all_occupation,))
	# all_data = all_data.transform(transform_function,axis=1)
	# all_data = transformNumeric(all_data)
	# return perform_data_split(all_data)
	return all_data

def getTrainingData():
	if os.path.isfile(training_data_path):
		result = pd.read_csv(training_data_path,sep='\t')
		return result
	filepath='ua.base'
	result = combine_data(filepath)
	result.to_csv(training_data_path,sep='\t',index=False)
	return result

def getTestData():
	if os.path.isfile(testing_data_path):
		result = pd.read_csv(testing_data_path,sep='\t')
		return result
	filepath='ua.test'
	result = combine_data(filepath)
	result.to_csv(testing_data_path,sep='\t',index=False)
	return result

def perform_data_split(all_data):
	training,testing  = train_test_split(all_data,test_size=0.15,random_state=40,shuffle=True)
	# save the data to csv
	training.to_csv(training_data_path,sep='\t',index=False)
	testing.to_csv(testing_data_path,sep='\t', index=False)
	return training

def load_genre_category():
	directory='datasets/ml-100k/u.genre'
	result={}
	all_genre={}
	with open(directory,'r') as fl:
		temp = [x.split('|') for x in fl.read().lower().split('\n') if x.strip()]
		for tm in temp:
			if len(tm)<2:
				continue
			result[tm[0]] =tm[1]
		return result

def load_occupation_category():
	directory='datasets/ml-100k/u.occupation'
	result={}
	all_genre={}
	with open(directory,'r') as fl:
		return [x for x in fl.read().split('\n') if x]

def combine_genre(all_data):
	#might have to pik the row one after the the other
	context_path='datasets/contextualdata_new.csv'
	contextual_data = pd.read_csv(context_path)
	contextual_data['CompanionContext'] = contextual_data['CompanionContext'].str.strip()
	all_data= pd.merge(all_data,contextual_data,left_on='movie_id',right_on='Movie_Id')
	all_data.insert(5,'music',0)
	all_data.insert(5,'family',0)
	all_data.insert(5,'biography',0)
	all_data.insert(5,'history',0)
	all_data.insert(5,'sport',0)
	all_data.insert(5,'short',0)
	genre_columns = ['short', 'sport','history', 'biography', 'family', 'music', 'unknown', 'action','adventure', 'animation', 'children', 'comedy', 'crime', 'documentary','drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery','romance', 'sci-fi', 'thriller', 'war', 'western']
	genre_lookup = load_genre_category()
	for index, row in all_data.iterrows():
		try:
			value=row['genre'].lower()
			values = [x.strip(' ,') for x in value.split(',')]
			all_data.at[index,values]=1
			# now update the genre to be just text
			temp=[]
			temp_genre = row[genre_columns]
			for tp in genre_columns:
				if row[tp]==1:
					temp.append(genre_lookup[tp])
			all_data.at[index,'genre']=','.join(temp)
		except Exception as e:
			all_data.at[index,['genre']]=''
			continue
	return all_data

def load_test_data():
	if os.path.isfile(testing_data_path):
		result=pd.read_csv(testing_data_path,sep='\t')
		return result
	return getTestData()
	

# print(getTrainingData())
# exit()
# all_data =combine_data()
# print(all_data.columns)
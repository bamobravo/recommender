"""
This file will contain a collection of function for manipulating the data
"""

import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
training_data_path='saved/training.csv'
testing_data_path = 'saved/testing.csv'

def combine_data():
	directory='datasets/ml-100k/'
	if os.path.isfile(training_data_path):
		result = pd.read_csv(training_data_path,sep='\t')
		return result
	u_data = pd.read_csv(directory+"u.data",sep='\t',names=['user_id','movie_id','rating','rating_time'])
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
	all_data['age_class']=all_data['age']/5
	all_data['rated']=all_data['rating']>3
	all_data = all_data.astype({'rated':'int'})
	return perform_data_split(all_data)
	return training

def perform_data_split(all_data):
	training,testing  = train_test_split(all_data,test_size=0.15,random_state=40,shuffle=True)
	# save the data to csv
	training.to_csv(training_data_path,sep='\t',index=False)
	testing.to_csv(testing_data_path,sep='\t', index=False)
	return training


def combine_genre(all_data):
	#might have to pik the row one after the the other
	context_path='datasets/contextualdata_new.csv'
	contextual_data = pd.read_csv(context_path)
	all_data= pd.merge(all_data,contextual_data,left_on='movie_id',right_on='Movie_Id')
	all_data.insert(5,'music',0)
	all_data.insert(5,'family',0)
	all_data.insert(5,'biography',0)
	all_data.insert(5,'history',0)
	all_data.insert(5,'sport',0)
	all_data.insert(5,'short',0)
	genre_columns = ['short', 'sport','history', 'biography', 'family', 'music', 'unknown', 'action','adventure', 'animation', 'children', 'comedy', 'crime', 'documentary','drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery','romance', 'sci-fi', 'thriller', 'war', 'western']
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
					temp.append(tp)
			all_data.at[index,'genre']=','.join(temp)
		except Exception as e:
			all_data.at[index,['genre']]=''
			continue
	return all_data

def load_test_data():
	result=pd.read_csv(testing_data_path,sep='\t')
	return result

# all_data =combine_data()
# print(all_data.columns)
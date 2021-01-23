"""
This file will contain a collection of function for manipulating the data
"""

import numpy as np
import pandas as pd 

def combine_data():
	directory='datasets/ml-100k/'
	u_data = pd.read_csv(directory+"u.data",sep='\t',names=['user_id','movie_id','rating','rating_time'])
	movie_genres=['unknown','action','adventure','animation','children','comedy','crime','documentary','drama','fantasy','film-noir','horror','musical','mystery','romance','sci-fi','thriller','war','western']
	movie_info=['movie_id','movie_title','release_date','video_release_date','IMDB_url']
	movie_header= movie_info+movie_genres
	u_movies=pd.read_csv(directory+"u.item",sep='|',names=movie_header)
	u_movies=combine_genre(u_movies)
	user_heading = ['user_id','age','gender','occupation','zip code']
	users_info = pd.read_csv(directory+"u.user",sep='|',names=user_heading)

	# now lets join the data together
	rating_movies = pd.merge(u_data,u_movies,on='movie_id')
	all_data = pd.merge(rating_movies,users_info,on='user_id')
	return all_data


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
	for index, row in all_data.iterrows():
		try:
			value=row['genre'].lower()
			values = [x.strip(' ,') for x in value.split(',')]
			all_data.at[index,values]=1
		except Exception as e:
			print(e)
			continue
	return all_data

all_data =combine_data()
print(all_data.columns)
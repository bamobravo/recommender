"""
This will be the main entry point for the project
"""

import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
import data

class Recommender:
	"""docstring """
	def __init__(self):
		self.variables=['zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
		self.buildModel()
		

	def buildModel(self):
		edges =[('movie_id','zip_code'),('movie_id','gender'),('movie_id','age_class'),('zip_code','occupation'),('gender','occupation'),('occupation','genre'),('age_class','occupation'),('genre','CompanionContext'),('CompanionContext','rated')]
		self.model = BayesianModel(edges)
		cpds = self.getCPDs()
		self.model.add_cpds(cpds)
		if not self.model.check_model():
			raise Exception('There is a problem creating the model')

	def getCPDs(self):
		genres = ['comedy','short', 'sport','history', 'biography', 'family', 'music', 'unknown', 'action','adventure', 'animation', 'children',  'crime', 'documentary','drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery','romance', 'sci-fi', 'thriller', 'war', 'western']
		all_data = data.combine_data()
		self.data = all_data[self.variables]
		mle = MaximumLikelihoodEstimator(self.model, self.data)
		straightForward = ['zip_code','gender','age_class','movie_id','occupation']
		cpds = []
		genre_cpd = self.get_genre_cpds(genres,self.data)
		companion_cpd = self.get_companion_context_cpd(genres,self.data)
		print(companion_cpd)
		exit()
		for item in straightForward:
			cpds.append(mle.estimate_cpd(item))
		print(cpds)
		exit()

	def get_genre_cpds(self,genre,data):
		total_count = len(genre)
		# build for each genre and for each of the movies
		result=[]
		occupations = list(pd.unique(data['occupation']))
		for ge in genre:
			occupation_count = {x:0 for x in occupations}
			temp_index = data['genre'].str.contains(ge,regex=False,case=False)
			occupation_data = data[temp_index]
			occupation_count,ct = self.estimate_genre_count(occupation_data,occupation_count)
			result.append((ge,occupation_count,ct))
		dt = [list(x[1].values()) for x in result]
		toReturn = TabularCPD(variable='genre',values=dt,variable_card=total_count,evidence=['occupation'],evidence_card=[len(occupations)])
		return toReturn

	def get_companion_context_cpd(self,genre,data):
		pass
	def estimate_genre_count(self,data,keys):
		total_count =0
		if data.empty:
			return keys,total_count
		for index,row in data.iterrows():
			occupation = row['occupation']
			keys[occupation]+=1
		for k in keys:
			keys[k]=keys[k]/data.shape[0]
		return keys,data.shape[0]


temp = Recommender()




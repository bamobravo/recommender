"""
This will be the main entry point for the project
"""

import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
import data
import random
import pickle
import utility
import os

class Recommender:
	"""docstring """
	def __init__(self):
		self.variables=['zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
		self.buildModel()
		

	def buildModel(self):
		filename = 'saved/model.bat'
		if os.path.isfile(filename):
			self.model = utility.loadModel()
			print('model loaded successfully')
			return

		edges =[('movie_id','zip_code'),('movie_id','gender'),('movie_id','age_class'),('zip_code','occupation'),('gender','occupation'),('occupation','genre'),('age_class','occupation'),('genre','CompanionContext'),('CompanionContext','rated')]
		self.model = BayesianModel(edges)
		zip_code,gender,age_class,movie_id,occupation,rated,genre_cpd,context_cpd = tuple(self.getCPDs())

		# genre_cpd,context_cpd = tuple(self.getCPDs())

		self.model.add_cpds(zip_code,gender,age_class,movie_id,occupation,rated,genre_cpd,context_cpd)
		# self.model.add_cpds(genre_cpd,context_cpd)
		if not self.model.check_model():
			raise Exception('There is a problem creating the model')
		utility.saveModel(self.model)
		print('model built successfully')


	def getCPDs(self):
		genres = ['comedy','short', 'sport','history', 'biography', 'family', 'music', 'unknown', 'action','adventure', 'animation', 'children',  'crime', 'documentary','drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery','romance', 'sci-fi', 'thriller', 'war', 'western']
		all_data = data.combine_data()
		self.data = all_data[self.variables]
		mle = MaximumLikelihoodEstimator(self.model, self.data)
		straightForward = ['zip_code','gender','age_class','movie_id','occupation','rated']
		cpds = []
		genre_cpd = self.get_genre_cpds(genres,self.data)
		companion_cpd = self.get_companion_context_cpd(genres,self.data)
		for item in straightForward:
			cpds.append(mle.estimate_cpd(item))
		cpds.append(companion_cpd)
		cpds.append(genre_cpd)
		
		return cpds

	def get_genre_cpds(self,genre,data):
		total_count = len(genre)
		# build for each genre and for each of the movies
		result=[]
		occupations = list(pd.unique(data['occupation']))
		for occupation in occupations:
			genre_count = {x:1 for x in genre}
			occupation_data = data[data['occupation']==occupation]
			genre_count,ct = self.estimate_genre_count(occupation_data,genre_count)
			result.append((occupation,genre_count,ct))
		dt = self.transpose([list(x[1].values()) for x in result])
		toReturn = TabularCPD(variable='genre',values=dt,variable_card=total_count,evidence=['occupation'],evidence_card=[len(occupations)])
		return toReturn

	def estimate_genre_count(self,data,keys):
		total_count =0
		total_count=len(keys.keys())

		for index,row in data.iterrows():
			genres = [ x.strip() for x in row['genre'].split(',') if x.strip()]
			if not genres:
				continue
			total_count+=len(genres)
			for g in genres:
				keys[g]+=1

		for k in keys:
			keys[k]=keys[k]/total_count
		#adjust to make probability exactly one
		diff = 1-sum(list(keys.values()))
		ind = random.randint(0,len(keys.keys())-1)
		keys[list(keys.keys())[ind]]+=diff
		return keys,total_count

	def get_companion_context_cpd(self,genre,data):
		context = list(pd.unique(data['CompanionContext']))
		res = []
		for ge in genre:
			context_count = {x:1 for x in context}
			ct_index = data['genre'].str.contains(ge,regex=False,case=False)
			ct_data = data[ct_index]
			# ct_data = data[data['CompanionContext']==ct]
			genre_count,ct = self.estimate_context_count(ct_data,context_count)
			res.append((ct,genre_count,ct))
		dt = self.transpose([list(x[1].values()) for x in res])
		toReturn = TabularCPD(variable='CompanionContext',values=dt,variable_card=len(context),evidence=['genre'],evidence_card=[len(genre)])
		return toReturn

	def estimate_context_count(self,data,keys):
		total_count =data.shape[0] + len(keys.keys())
		for index,row in data.iterrows():
			ct = row['CompanionContext']
			keys[ct]+=1
		for k in keys:
			keys[k]=keys[k]/total_count
		diff = 1-sum(list(keys.values()))
		ind = random.randint(0,len(keys.keys())-1)
		keys[list(keys.keys())[ind]]+=diff
		return keys,total_count


	def transpose(self,lst):
		arr = np.array(lst)
		result=np.transpose(arr)
		return result


	# the set of functions for performing inference on the graphical model

	def queryModel(self,variables,evidence):
		"""
			The query should be a set of parameters and a set of values
		"""
		return self.model.query(variables=variables,evidence=evidence)
		pass

temp = Recommender()




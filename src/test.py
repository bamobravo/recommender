import data
import utility
from itertools import product
import pandas as pd
from pgmpy.inference import VariableElimination
import re


def start_training():
	testData = data.getTestData()
	model = utility.loadModel()
	y_values=testData['rated']
	x_values = testData.loc[:,]
	correct=0
	evidence_columns = ['age','gender','occupation','zip_code','CompanionContext']
	for index,row in testData.iterrows():
		expected = row['rated']
		variable =['rated']
		evidences ={x:row[x] for x in evidence_columns}
		derived = model.query(variables=variables,evidence=evidences)
		print(derived)
		exit()

# start_training()

def transform_data():
	pass

def groupby(field,data):
	uniqueValues = data[field].unique()
	result ={}
	min_row=3
	for id in uniqueValues:
		temp = data[data[field]==id]
		if temp.shape[0] < min_row:
			continue
		key = str(id)
		result[key]=temp
	return result

def getRecommendationContext(groupedData):
	model = utility.loadModel()
	inference = VariableElimination(model)
	testResult={}
	for user in groupedData:
		try:
			temp_data = groupedData[user]
			user_fields=['rated','CompanionContext']
			variable=['movie_id']
			user_field_values = [1,int(user)]
			evidences={x[0]:x[1] for x in zip(user_fields,user_field_values)}
			res=inference.query(variables=variable,evidence=evidences)
			result =(x for x in zip(res.state_names['movie_id'],res.values) if x[1])
			result = sorted(result,key=lambda x:x[1],reverse=True)
			testResult[user]=result
		except Exception as e:

			print(e)
			print(evidences)
			print('skipped')
			continue
	return testResult
def getRecommendation(groupedData):
	model = utility.loadModel()
	inference = VariableElimination(model)
	testResult={}
	for user in groupedData:
		try:
			temp_data = groupedData[user]
			user_fields=['gender','zip_code','age_class','rated','CompanionContext','occupation']
			variable=['movie_id']
			user_field_values = [temp_data.iloc[0]['gender'],temp_data.iloc[0]['zip_code'],temp_data.iloc[0]['age_class'],1,temp_data.iloc[0]['CompanionContext'],temp_data.iloc[0]['occupation']]
			evidences={x[0]:x[1] for x in zip(user_fields,user_field_values)}
			res=inference.query(variables=variable,evidence=evidences)
			result =(x for x in zip(res.state_names['movie_id'],res.values) if x[1])
			result = sorted(result,key=lambda x:x[1],reverse=True)
			testResult[user]=result
		except Exception as e:

			print(e)
			print(evidences)
			print('skipped')
			continue
	return testResult

def getRecommendationByGenre(groupedData):
	model = utility.loadModel()
	inference = VariableElimination(model)
	testResult={}
	for user in groupedData:
		try:
			temp_data = groupedData[user]
			user_fields=['genre','rated']
			variable=['movie_id']
			user_field_values = [int(user),1]
			evidences={x[0]:x[1] for x in zip(user_fields,user_field_values)}
			res=inference.query(variables=variable,evidence=evidences)
			result =(x for x in zip(res.state_names['movie_id'],res.values) if x[1])
			result = sorted(result,key=lambda x:x[1],reverse=True)
			testResult[user]=result
		except Exception as e:

			print(e)
			print('skipped')
			continue
	return testResult

# this is the function that will perform data grouping by user and by gender
def testByUser():
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data()
	testData = testData[important]
	testData = testData.dropna()
	groupedData = groupby('user_id',testData)
	all_movies_recommendation = getRecommendation(groupedData)
	total_number =0
	correct=0
	for user in all_movies_recommendation:
		testData=groupedData[user]['movie_id'].unique()
		predictedData = [x[0] for x in all_movies_recommendation[user]]
		correct_value = [x for x in testData  if x in predictedData]
		total_number+= len(testData)
		correct += len(correct_value)
	accuracy = correct/total_number
	print('The accuracy ',accuracy)


def testByContext():
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data()
	testData = testData[important]
	testData = testData.dropna()
	groupedData = groupby('CompanionContext',testData)
	all_movies_recommendation = getRecommendationContext(groupedData)
	total_number =0
	correct=0
	for user in all_movies_recommendation:
		testData=groupedData[user]['movie_id'].unique()
		predictedData = [x[0] for x in all_movies_recommendation[user]]
		correct_value = [x for x in testData  if x in predictedData]
		total_number+= len(testData)
		correct += len(correct_value)
	accuracy = correct/total_number
	print('The accuracy ',accuracy)




def groupByGenre(testData):
	all_genre = data.load_genre_category()
	result={}
	for genre in all_genre:
		ge = all_genre[genre]
		index = testData['genre'].apply(data.hasGenre,args=(ge,))
		temp =testData[index]
		if temp.empty:
			continue
		all_movies=list(temp['movie_id'].unique())
		result[ge]=all_movies
	return result

def testByGenre(type=1):
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data()
	testData = testData[important]
	testData = testData.dropna()
	groupedData = groupByGenre(testData)
	all_movies_recommendation = getRecommendationByGenre(groupedData)
	total_number =0
	correct=0
	for user in all_movies_recommendation:
		movies=groupedData[user]
		predictedData = [x[0] for x in all_movies_recommendation[user]]
		correct_value = [x for x in movies  if x in predictedData]
		total_number+= len(movies)
		correct += len(correct_value)
	accuracy = correct/total_number
	print('The accuracy ',accuracy)


def runQueries():
	print('runing the first query with test')
	print("\t\tRecommendation of Top Movies based on given movie title genre")
	testByGenre()
	print('\n\n\n')

	print("Running the second query with test")
	print('\t\t Recommendation based on movies user have watched')
	testByGenre()
	print('\n\n\n')

	print("Running the third query with test")
	print('\t\t Recommendation using movie similarity')
	testByGenre()
	print('\n\n\n')

	print("Running the fourth query with test")
	print('\t\t Recommendation using user similarity')
	testByUser()
	print('\n\n\n')

	print("Running the fifth query with test")
	print('\t\t Recommendation using context')
	testByContext()
	print('\n\n\n')


runQueries()
# testByGenre()
# grouping = testByUser()
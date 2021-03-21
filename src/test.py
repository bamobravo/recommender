import data
import utility
from itertools import product
import pandas as pd
from pgmpy.inference import VariableElimination
import re
from main import Recommender
import traceback

"""
Most of the function in this file are just preparing the test condition and also the data for testing
"""

# def start_training():
# 	testData = data.getTestData()
# 	model = utility.loadModel()
# 	y_values=testData['rated']
# 	x_values = testData.loc[:,]
# 	correct=0
# 	evidence_columns = ['age','gender','occupation','zip_code','CompanionContext']
# 	for index,row in testData.iterrows():
# 		expected = row['rated']
# 		variable =['rated']
# 		evidences ={x:row[x] for x in evidence_columns}
# 		derived = model.query(variables=variables,evidence=evidences)
# 		print(derived)
# 		exit()

# start_training()


def groupby(field,data):
	'''
		This is a generic testing data grouping function,the function help categorise test data based on a particular field
	'''
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

def getRecommendationContext(groupedData,fold):
	model = utility.loadModel(fold)
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

def getRecommendation(groupedData,fold):
	"""
	This function get recommendation given an evidence variable.
	"""
	model = utility.loadModel(fold)
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

def getRecommendationByGenre(groupedData,fold):
	"""
	This is also a data manipulation function for grouping the test data based on movie genre
	"""

	model = utility.loadModel(fold)
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
def testByUser(fold=False):
	"""
	This function perform test based on user characteristics
	"""
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data(fold)
	testData = testData[important]
	testData = testData.dropna()
	groupedData = groupby('user_id',testData)
	all_movies_recommendation = getRecommendation(groupedData,fold)
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
	return accuracy


def testByContext(fold=False):
	'''
		perform test based on companion context
	'''
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data()
	testData = testData[important]
	testData = testData.dropna()
	groupedData = groupby('CompanionContext',testData)
	all_movies_recommendation = getRecommendationContext(groupedData,fold)
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
	return accuracy


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

def testByGenre(type=1,fold=False):
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data(fold)
	testData = testData[important]
	testData = testData.dropna()
	groupedData = groupByGenre(testData)
	all_movies_recommendation = getRecommendationByGenre(groupedData,fold)
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
	return accuracy

#Testing functions

def test1():
	print('runing the first query with test')
	print("\t\tRecommendation of Top Movies based on given movie title genre")
	fold_count = 5
	total_accuracy=0
	for i in range(fold_count):
		fold = i+1
		temp = Recommender(fold)
		tempModel = temp.buildModel()
		accuracy=testByGenre(fold=fold)
		total_accuracy+=accuracy
		print(accuracy)
	print(total_accuracy/fold_count)
	print('\n\n\n')

def test2():
	print("Running the second query with test")
	print('\t\t Recommendation based on movies user have watched')
	fold_count = 5
	total_accuracy=0
	for i in range(fold_count):
		fold = i+1
		temp = Recommender(fold)
		tempModel = temp.buildModel()
		accuracy=testByGenre(fold=fold)
		total_accuracy+=accuracy
		print(accuracy)
	print(total_accuracy/fold_count)
	print('\n\n\n')

def test3():
	print("Running the third query with test")
	print('\t\t Recommendation using movie similarity')
	fold_count = 5
	total_accuracy=0
	for i in range(fold_count):
		fold = i+1
		temp = Recommender(fold)
		tempModel = temp.buildModel()
		accuracy=testByGenre(fold=fold)
		total_accuracy+=accuracy
		print(accuracy)
	print(total_accuracy/fold_count)
	print('\n\n\n')

def test4():
	print("Running the fourth query with test")
	print('\t\t Recommendation using user similarity')
	fold_count = 5
	total_accuracy=0
	for i in range(fold_count):
		fold = i+1
		temp = Recommender(fold)
		tempModel = temp.buildModel()
		accuracy=testByUser(fold=fold)
		total_accuracy+=accuracy
		print(accuracy)
	print(total_accuracy/fold_count)
	print('\n\n\n')

def test5():
	print("Running the fifth query with test")
	print('\t\t Recommendation using context')
	fold_count = 5
	total_accuracy=0
	for i in range(fold_count):
		fold = i+1
		temp = Recommender(fold)
		tempModel = temp.buildModel()
		accuracy=testByContext(fold=fold)
		total_accuracy+=accuracy
		print(accuracy)
	print(total_accuracy/fold_count)
	print('\n\n\n')

def runQueries():
	"""
	Entry to testing
	"""
	tests = [test1,test2,test3,test4,test5]
	try:
		test_number =int(input('Kindly specify the test you will like to run (1-5)'))
		test_to_run = tests[test_number-1]
		test_to_run()
	except Exception as e:
		print(e)
		print(traceback.format_exc())
		print('invalid input')
	

	

	

	

	

# call the main function
runQueries()
# testByGenre()
# grouping = testByUser()

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

def getRecommendationContext(model,groupedData,fold):
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

def getRecommendation(model,groupedData,fold):
	"""
	This function get recommendation given an evidence variable.
	"""
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

def getRecommendationByGenre(model,groupedData,fold):
	"""
	This is also a data manipulation function for grouping the test data based on movie genre
	"""

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
	model = utility.loadModel(fold)
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data(fold)
	testData = testData[important]
	testData = testData.dropna()
	testData['genre'] = testData['genre'].astype('int')
	groupedData = groupby('user_id',testData)
	precision, recall,fscore = estimateMetrics(model,testData)
	all_movies_recommendation = getRecommendation(model,groupedData,fold)
	total_number =0
	correct=0
	for user in all_movies_recommendation:
		testData=groupedData[user]['movie_id'].unique()
		predictedData = [x[0] for x in all_movies_recommendation[user]]
		correct_value = [x for x in testData  if x in predictedData]
		total_number+= len(testData)
		correct += len(correct_value)
	accuracy = correct/total_number
	print('The accuracy ',accuracy,' precision is:',precision,' recall is: ',recall,' fscore is: ',fscore)
	return accuracy,precision,recall,fscore


def testByContext(fold=False):
	'''
		perform test based on companion context
	'''
	model = utility.loadModel(fold)
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data()
	testData = testData[important]
	testData = testData.dropna()
	testData['genre'] = testData['genre'].astype('int')
	groupedData = groupby('CompanionContext',testData)
	precision, recall,fscore = estimateMetrics(model,testData)
	all_movies_recommendation = getRecommendationContext(model,groupedData,fold)
	total_number =0
	correct=0
	for user in all_movies_recommendation:
		testData=groupedData[user]['movie_id'].unique()
		predictedData = [x[0] for x in all_movies_recommendation[user]]
		correct_value = [x for x in testData  if x in predictedData]
		total_number+= len(testData)
		correct += len(correct_value)
	accuracy = correct/total_number
	print('The accuracy ',accuracy,' precision is:',precision,' recall is: ',recall,' fscore is: ',fscore)
	return accuracy,precision,recall,fscore

def groupByGenreForMetrics(testData):
	all_genre = data.load_genre_category()
	result={}
	for genre in all_genre:
		ge = all_genre[genre]
		index = testData['genre']==int(ge)
		temp =testData[index]
		if temp.empty:
			continue
		# all_movies=temp['movie_id']
		result[ge]=temp
	return result

def groupByGenre(testData):
	all_genre = data.load_genre_category()
	result={}
	for genre in all_genre:
		ge = all_genre[genre]
		index = testData['genre']==int(ge)
		temp =testData[index]
		if temp.empty:
			continue
		all_movies=list(temp['movie_id'].unique())
		result[ge]=all_movies
	return result

def testByGenre(type=1,fold=False):
	model = utility.loadModel(fold)
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data(fold)
	testData = testData[important]
	testData = testData.dropna()
	testData['genre'] = testData['genre'].astype('int')
	groupedData = groupByGenre(testData)
	precision, recall,fscore = estimateMetrics(model,testData)
	all_movies_recommendation = getRecommendationByGenre(model,groupedData,fold)
	total_number =0
	correct=0 
	for user in all_movies_recommendation:
		movies=groupedData[user]
		predictedData = [x[0] for x in all_movies_recommendation[user]]
		correct_value = [x for x in movies  if x in predictedData]
		total_number+= len(movies)
		correct += len(correct_value)
	accuracy = correct/total_number
	print('The accuracy ',accuracy,' precision is:',precision,' recall is: ',recall,' fscore is: ',fscore)
	return accuracy,precision,recall,fscore

#Testing functions


def estimateSingleMetrics(inference,data):
	columns=['movie_id','genre','gender','CompanionContext','zip_code','occupation','age_class']
	y_values = data['rated']
	tPositive=0
	tNegative =0
	fPositive=0
	fNegative =0
	for index, row in data.iterrows():
		evidences={x:row[x] for x in columns}
		variable =['rated']
		result = inference.map_query(variables=variable,evidence=evidences)
		expected = row['rated']
		predicted = result['rated']
		if predicted==1:
			if expected==1:
				tPositive+=1
			else:
				fPositive+=1
		else:
			if expected==1:
				fPositive+=1
			else:
				fNegative+=1
		# exit()
	precision = tPositive/(tPositive+fPositive)
	recall =  tPositive/(tPositive+fNegative)
	fscore = (2* precision * recall)/(precision+recall)
	return precision,recall,fscore

def estimateMetrics(model,data):
	data = groupByGenreForMetrics(data)
	inference = VariableElimination(model)
	precision =0
	recall =0
	fscore = 0
	count=0
	for genre in data:
		p,r,f =estimateSingleMetrics(inference,data[genre])
		precision+=p
		recall+=r
		fscore+=f
		count+=1
	return precision/count,recall/count,fscore/count

def test1():
	print('running the first query with test')
	print("\t\tRecommendation of Top Movies based on given movie title genre")
	fold_count = 5
	total_accuracy=0
	total_precision=0
	total_recall=0
	total_fscore=0
	for i in range(fold_count):
		fold = i+1
		temp = Recommender(fold)
		tempModel = temp.buildModel()
		accuracy,precision,recall,fscore=testByGenre(fold=fold)
		total_accuracy+=accuracy
		total_precision+=precision
		total_recall+=recall
		total_fscore+=fscore
		# print(accuracy)
	print('Accuracy: ',total_accuracy/fold_count)
	print('Precision: ',total_precision/fold_count)
	print('Recall: ',total_recall/fold_count)
	print('F-score: ',total_fscore/fold_count)
	print('\n\n\n')

def test2():
	print("Running the second query with test")
	print('\t\t Recommendation based on movies user have watched')
	fold_count = 5
	total_accuracy=0
	total_precision=0
	total_recall=0
	total_fscore=0
	for i in range(fold_count):
		fold = i+1
		temp = Recommender(fold)
		tempModel = temp.buildModel()
		accuracy,precision,recall,fscore=testByGenre(fold=fold)
		total_accuracy+=accuracy
		total_precision+=precision
		total_recall+=recall
		total_fscore+=fscore
	print('Accuracy: ',total_accuracy/fold_count)
	print('Precision: ',total_precision/fold_count)
	print('Recall: ',total_recall/fold_count)
	print('F-score: ',total_fscore/fold_count)
	print('\n\n\n')

def test3():
	print("Running the third query with test")
	print('\t\t Recommendation using movie similarity')
	fold_count = 5
	total_accuracy=0
	total_precision=0
	total_recall=0
	total_fscore=0
	for i in range(fold_count):
		fold = i+1
		temp = Recommender(fold)
		tempModel = temp.buildModel()
		accuracy,precision,recall,fscore=testByGenre(fold=fold)
		total_accuracy+=accuracy
		total_precision+=precision
		total_recall+=recall
		total_fscore+=fscore
	print('Accuracy: ',total_accuracy/fold_count)
	print('Precision: ',total_precision/fold_count)
	print('Recall: ',total_recall/fold_count)
	print('F-score: ',total_fscore/fold_count)
	print('\n\n\n')

def test4():
	print("Running the fourth query with test")
	print('\t\t Recommendation using user similarity')
	fold_count = 5
	total_accuracy=0
	total_precision=0
	total_recall=0
	total_fscore=0
	for i in range(fold_count):
		fold = i+1
		temp = Recommender(fold)
		tempModel = temp.buildModel()
		accuracy,precision,recall,fscore=testByUser(fold=fold)
		total_accuracy+=accuracy
		total_precision+=precision
		total_recall+=recall
		total_fscore+=fscore
		# print(accuracy)
	print('Accuracy: ',total_accuracy/fold_count)
	print('Precision: ',total_precision/fold_count)
	print('Recall: ',total_recall/fold_count)
	print('F-score: ',total_fscore/fold_count)
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
		accuracy,precision,recall,fscore=testByContext(fold=fold)
		total_accuracy+=accuracy
		total_precision+=precision
		total_recall+=recall
		total_fscore+=fscore
		# print(accuracy)
	print('Accuracy: ',total_accuracy/fold_count)
	print('Precision: ',total_precision/fold_count)
	print('Recall: ',total_recall/fold_count)
	print('F-score: ',total_fscore/fold_count)
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

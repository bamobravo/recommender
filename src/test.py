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

fold_count = 2
#this is the value that test if to get best N count of more


def translateFold(fold):
	temp =['a','b']
	return temp[fold-1]

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
			user_fields=['gender','zip_code','age_class','rated','CompanionContext']
			variable=['movie_id']
			user_field_values = [temp_data.iloc[0]['gender'],temp_data.iloc[0]['zip_code'],temp_data.iloc[0]['age_class'],1,temp_data.iloc[0]['CompanionContext'],temp_data.iloc[0]['occupation']]
			evidences={x[0]:x[1] for x in zip(user_fields,user_field_values)}
			res=inference.query(variables=variable,evidence=evidences)
			result =(x for x in zip(res.state_names['movie_id'],res.values) if x[1])
			result = sorted(result,key=lambda x:x[1],reverse=True)
			testResult[user]=result
			# this is to quickly test what is happening and how to quickly perform the top testing
			# return testResult
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

def filterAbsentMovies(model, data):
	minOccurrence=50
	presents =model.get_cpds('movie_id').state_names['movie_id']
	result = data[data['movie_id'].isin(presents)]
	counting = result.groupby('movie_id')
	include =[]
	for name,group in counting:
		if group.shape[0] >= minOccurrence:
			include.append(name)
	result = result[result['movie_id'].isin(include)]
	return result
	
# this is the function that will perform data grouping by user and by gender
def testByUser(fold=False):
	"""
	This function perform test based on user characteristics
	"""
	global topN
	# fold = translateFold(fold)
	model = utility.loadModel(fold)
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data(fold)
	testData = testData[important]
	testData = testData.dropna()
	testData = filterAbsentMovies(model,testData)
	testData['genre'] = testData['genre'].astype('int')
	groupedData = groupby('user_id',testData)
	precision,recall,fscore=0,0,0
	if topN < 0:
		precision, recall,fscore = estimateMetrics(model,testData)

	all_movies_recommendation = getRecommendation(model,groupedData,fold)
	#you can change to top N here
	if topN>0:
		# all_movies_recommendation = 
		precision,recall,fscore = estimateTopNMetrics(model,testData,all_movies_recommendation)
	total_number =0
	correct=0
	for user in all_movies_recommendation:
		tempData = None
		if topN>0:
			tempData=groupedData[user][:topN]['movie_id'].unique()
		else:
			tempData=groupedData[user]['movie_id'].unique()
		predictedData = [x[0] for x in all_movies_recommendation[user]]
		# this should be data that was part of the data the model was trained with
		correct_value = [x for x in tempData  if x in predictedData]
		total_number+= len(tempData)
		correct += len(correct_value)
	accuracy = correct/total_number if total_number > 0 else False
	print('The accuracy ',accuracy,' precision is:',precision,' recall is: ',recall,' fscore is: ',fscore)
	return accuracy,precision,recall,fscore


def testByContext(fold=False):
	'''
		perform test based on companion context
	'''
	# fold = translateFold(fold)
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
	# fold = translateFold(fold)
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
		# is the movies part of the one used in generating the models?
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
	columns=['movie_id','rated','age_class']
	y_values = data['gender']
	tPositive=0
	tNegative =0
	fPositive=0
	fNegative =0
	count=0
	for index, row in data.iterrows():
		try:
			evidences={x:row[x] for x in columns}
			variable =['gender']
			result = inference.map_query(variables=variable,evidence=evidences)
			count+=1
			expected = row['gender']
			predicted = result['gender']
			if predicted=='M':
				if expected==predicted:
					tPositive+=1
				else:
					fPositive+=1
			else:
				if expected==predicted:
					tNegative+=1
				else:
					fNegative+=1
			print('processed ',count,' of ',data.shape[0])
			print("True Positive:"+str(tPositive),"False Positive: "+str(fPositive),"False Negative: "+str(fNegative),"True Negative:"+str(tNegative),sep='\t')
			print('\n')
			# exit()
		except Exception as e:
			print(e)
			print('skipping some functions here')
			continue
	precision = tPositive/(tPositive+fPositive) if (tPositive+fPositive) > 0 else False
	recall =  tPositive/(tPositive+fNegative) if (tPositive+fNegative)>0 else False
	fscore = 2*(( precision * recall)/(precision+recall)) if (precision+recall) > 0 else False
	print('completed single iteration \n\n\n')
	print(precision,recall,fscore)
	return precision,recall,fscore

def estimateMetrics(model,data):
	print('estimating new metrics \n\n')
	data = groupByGenreForMetrics(data)
	inference = VariableElimination(model)
	precision =0
	recall =0
	fscore = 0
	count=0
	for genre in data:
		print('estimating metrics for genre ',genre,' total count',data[genre].shape[0],'\n\n----------------------------------------------------------\n\n')
		p,r,f =estimateSingleMetrics(inference,data[genre])
		if p and r and f:
			precision+=p
			recall+=r
			fscore+=f
			count+=1
	# if count ==0:
	# 	return False,False,False
	print('this is the first thing to note')
	print(count)
	print('printing the value of count here', count,'\n\n\n')
	return precision/count,recall/count,fscore/count

def filterTopNMovies(data,recommendations):
	global topN
	distinctMovies = []
	for user in recommendations:
		# get just the movie id
		temp =[x[0] for x in recommendations[user][:topN]]
		distinctMovies+=temp
	distinctMovies = set(distinctMovies)
	result = data[data['movie_id'].isin(distinctMovies)]
	return result

def estimateTopNMetrics(model,all_data,movie_recommendations):
	print('estimating new metrics \n\n')
	data = filterTopNMovies(all_data,movie_recommendations)
	data = groupByGenreForMetrics(data)
	inference = VariableElimination(model)
	precision =0
	recall =0
	fscore = 0
	count=0
	for genre in data:
		print('estimating metrics for genre ',genre,' total count',data[genre].shape[0],'\n\n----------------------------------------------------------\n\n')
		p,r,f =estimateSingleMetrics(inference,data[genre])
		precision+=p
		recall+=r
		fscore+=f
		count+=1
	return precision/count,recall/count,fscore/count

def test1():
	print('running the first query with test')
	print("\t\tRecommendation of Top Movies based on given movie title genre")
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
	print(topN)
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
	total_accuracy=0
	total_precision=0
	total_recall=0
	total_fscore=0
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


topN=-1
def runQueries():
	"""
	Entry to testing
	split the training data before trying to running the split
	"""
	foldCount=5
	global topN
	tests = [test1,test2,test3,test4,test5]
	try:
		test_number =int(input('Kindly specify the test you will like to run (1-5)'))
		test_to_run = tests[test_number-1]
		temp = input('What value of N will do you like to run for the top N test? (leave empty of type -1 to skip top N test)')
		if not temp:
			temp='-1'
		if temp.isnumeric():
			topN= int(temp)
			print(topN)
		test_to_run()
	except Exception as e:
		print(e)
		print(traceback.format_exc())
		print('invalid input')
	

	

	

	

	

# call the main function
runQueries()
# testByGenre()
# grouping = testByUser()

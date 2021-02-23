import data
import utility
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling


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

# this is the function that will perform data grouping by user and by gender
def testByUser():
	important =['user_id','zip_code','occupation','gender','age_class','movie_id','genre','CompanionContext','rated']
	testData = data.load_test_data()
	testData = testData[important]
	testData = testData.dropna()
	model = utility.loadModel()
	inference = VariableElimination(model)
	groupedData = groupby('user_id',testData)
	testResult=[]
	for user in groupedData:
		try:
			temp_data = groupedData[user]
			user_fields=['gender','zip_code','age_class','rated','CompanionContext']
			variable=['movie_id']
			user_field_values = [temp_data.iloc[0]['gender'],temp_data.iloc[0]['zip_code'],temp_data.iloc[0]['age_class'],1,temp_data.iloc[0]['CompanionContext']]
			evidences={x[0]:x[1] for x in zip(user_fields,user_field_values)}
			res=inference.query(variables=variable,evidence=evidences)
			print(res)
			exit()
		except Exception as e:
			print(e)
			print('skipped')
			continue
		

def getRecommendedMovie(model,evidences):
	inference = VariableElimination(model)

		# now extract users characteristics

	print(groupedData)
	exit()

def getByGenre():
	pass

grouping = testByUser()
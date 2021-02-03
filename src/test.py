import data
import utility


def start_training():
	testData = data.load_test_data()
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

start_training()
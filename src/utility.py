import pickle

def saveModel(model,filename):
	# filename='saved/model.bat'
	try:
		with open(filename,'wb') as fl:
			pickle.dump(model,fl)
	except Exception as e:
		# just incase i want to  handle the exception here
		raise e


def loadModel(filename):
	# filename='saved/model.bat'
	if isinstance(filename,int) or len(filename) ==1:
		filename = 'saved/model'+str(filename)+'.bat'
	try:
		with open(filename,'rb') as fl:
			result = pickle.load(fl)
		return result
	except Exception as e:
		raise e
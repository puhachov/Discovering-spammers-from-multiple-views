import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

class singleview():

	def __init__(self, data, class_):
		self.X              = np.copy(np.transpose(data))
		self.ground_truth   = np.sum(class_, axis = 1)


	def evaluate(self, model, training_size):

		X_train, X_test, y_train, y_test = train_test_split(self.X, self.ground_truth)	
		clf                 = model.fit(X_train, y_train)
		y_pred              = clf.predict(X_test)
		confusion_matrix_   = confusion_matrix(y_test, y_pred)
		precision           = confusion_matrix_[0,0]/(confusion_matrix_[0,0] + confusion_matrix_[0,1])
		recall              = confusion_matrix_[0,0]/(confusion_matrix_[0,0] + confusion_matrix_[1,0])
		F1_score            = 2*precision*recall/(precision + recall)
		confusion_matrix_df = pd.DataFrame(data    = confusion_matrix_,
                        				   columns = ['Actual_Spammer', 'Actual_Legitimate'],
                        				   index   = ['Predicted_Spammer ','Predicted_Legitimate'])
                        				
		print("Precision: {}\n".format(precision))
		print("Recall: {}\n".format(recall))
		print("F1-score: {}\n".format(F1_score))
		print("Confusion Matrix:\n {}\n".format(confusion_matrix_))
		return precision, recall, F1_score, confusion_matrix_





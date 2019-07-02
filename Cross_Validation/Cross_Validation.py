
from sklearn.model_selection import cross_validate
scoring =  ['accuracy','precision_macro', 'recall_macro']

'--------------------------------------------------------------------------------------------------------------------'
'-------------------------------------------------------------------------------------------------------------------'

Prediction= cross_validate(classifier, X, y, scoring=scoring, cv = 10)

sorted(Prediction.keys())

Build_time = Prediction['fit_time'].sum()
Build_time

Test_time = Prediction['score_time'].sum()
Test_time 

Accuracy = Prediction['test_accuracy'].mean() * 100
Accuracy

Precision = Prediction['test_precision_macro'].mean() * 100
Precision

Recall = Prediction['test_recall_macro'].mean() * 100
Recall


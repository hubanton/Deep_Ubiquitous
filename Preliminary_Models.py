from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from Dataloading import get_patient_data
import lazypredict
from lazypredict.Supervised import LazyClassifier

X, y = get_patient_data(use_interpolation=False)

X = X.drop(['file_name', 'frame_id'], axis=1)
y = y.drop(['file_name', 'frame_id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

sm = SMOTE(random_state=42)
X_res_train, y_res_train = sm.fit_resample(X_train, y_train)
X_res_train = X_res_train.to_numpy()
y_res_train = y_res_train.to_numpy()

clf = LazyClassifier(predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

models.to_csv("Model_results.csv")

print(models)

# print("Results on Linear Regression: \n")
#
# reg = LinearRegression().fit(X_train, y_train)
# result = reg.predict(X_test)
#
# result = (result > 0.5) * 1
#
# print("patient data")
# print(f"Accuracy: {accuracy_score(y_test, result)}")
# print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, result)}")
# print(f"F1-score: {f1_score(y_test, result)}")
# print(f"Confusion matrix:\n {confusion_matrix(y_test, result)}")
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# reg = LinearRegression().fit(X_res_train, y_res_train)
# result = reg.predict(X_test)
#
# result = (result > 0.5) * 1
#
# print("\npatient data extended with SMOTE")
# print(f"Accuracy: {accuracy_score(y_test, result)}")
# print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, result)}")
# print(f"F1-score: {f1_score(y_test, result)}")
# print(f"Confusion matrix:\n {confusion_matrix(y_test, result)}")
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# print("\n\nResults on SVM: \n")
#
# reg = SVC().fit(X_train, y_train)
# result = reg.predict(X_test)
#
# result = (result > 0.5) * 1
#
# print("patient data")
# print(f"Accuracy: {accuracy_score(y_test, result)}")
# print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, result)}")
# print(f"F1-score: {f1_score(y_test, result)}")
# print(f"Confusion matrix:\n {confusion_matrix(y_test, result)}")
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# reg = SVC().fit(X_res_train, y_res_train)
# result = reg.predict(X_test)
#
# result = (result > 0.5) * 1
#
# print("\npatient data extended with SMOTE")
# print(f"Accuracy: {accuracy_score(y_test, result)}")
# print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, result)}")
# print(f"F1-score: {f1_score(y_test, result)}")
# print(f"Confusion matrix:\n {confusion_matrix(y_test, result)}")
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# print("\n\nResults on Random Forest: \n")
#
# reg = RandomForestClassifier().fit(X_train, y_train)
# result = reg.predict(X_test)
#
# result = (result > 0.5) * 1
#
# print("patient data")
# print(f"Accuracy: {accuracy_score(y_test, result)}")
# print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, result)}")
# print(f"F1-score: {f1_score(y_test, result)}")
# print(f"Confusion matrix:\n {confusion_matrix(y_test, result)}")
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# reg = RandomForestClassifier().fit(X_res_train, y_res_train)
# result = reg.predict(X_test)
#
# result = (result > 0.5) * 1
#
# print("\npatient data extended with SMOTE")
# print(f"Accuracy: {accuracy_score(y_test, result)}")
# print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, result)}")
# print(f"F1-score: {f1_score(y_test, result)}")
# print(f"Confusion matrix:\n {confusion_matrix(y_test, result)}")
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# print("\n\nResults on Nearest Neighbor: \n")
#
# reg = KNeighborsClassifier().fit(X_train, y_train)
# result = reg.predict(X_test)
#
# result = (result > 0.5) * 1
#
# print("patient data")
# print(f"Accuracy: {accuracy_score(y_test, result)}")
# print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, result)}")
# print(f"F1-score: {f1_score(y_test, result)}")
# print(f"Confusion matrix:\n {confusion_matrix(y_test, result)}")
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# reg = KNeighborsClassifier().fit(X_res_train, y_res_train)
# result = reg.predict(X_test)
#
# result = (result > 0.5) * 1
#
# print("\npatient data extended with SMOTE")
# print(f"Accuracy: {accuracy_score(y_test, result)}")
# print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, result)}")
# print(f"F1-score: {f1_score(y_test, result)}")
# print(f"Confusion matrix:\n {confusion_matrix(y_test, result)}")

# ----------------------------------------------------------------------------------------------------------------------



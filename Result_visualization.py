import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dataframe with the results
df = pd.DataFrame({'Algorithm': ['Linear Regression', 'Linear Regression', 'SVM', 'SVM', 'Random Forest', 'Random Forest', 'Nearest Neighbor', 'Nearest Neighbor'],
                   'Sample': ['Patient Data', 'Patient Data SMOTE', 'Patient Data', 'Patient Data SMOTE', 'Patient Data', 'Patient Data SMOTE', 'Patient Data', 'Patient Data SMOTE'],
                   'Accuracy': [0.9707174231332357, 0.7389192067083722, 0.9804339145481166, 0.7736589910821243, 0.9916145348063357, 0.9900173033408758, 0.9894848928523892, 0.9783708239052309],
                   'Balanced Accuracy': [0.5, 0.8467863019333608, 0.6692153434800494, 0.866884341149047, 0.8689411079116962, 0.9166101055806939, 0.8469045660222131, 0.896284108048814],
                   'F1-score': [0.0, 0.17739568043615017, 0.5033783783783783, 0.19995295224653023, 0.8376288659793815, 0.8310810810810813, 0.7948051948051948, 0.6865959498553519]})

# Create a bar plot of accuracy for each algorithm
plt.figure(figsize=(10,6))
sns.barplot(x='Algorithm', y='Accuracy', hue='Sample', data=df)
plt.title('Accuracy of Each Algorithm')
plt.legend(loc='lower right')
plt.ylim([0.6, 1])
plt.ylabel('Accuracy')
plt.xlabel('Algorithm')
plt.show()

# Create a bar plot of balanced accuracy for each algorithm
plt.figure(figsize=(10,6))
sns.barplot(x='Algorithm', y='Balanced Accuracy', hue='Sample', data=df)
plt.title('Balanced Accuracy of Each Algorithm')
plt.legend(loc='lower right')
plt.ylim([0.4, 1])
plt.ylabel('Balanced Accuracy')
plt.xlabel('Algorithm')
plt.show()

# Create a bar plot of F1-score for each algorithm
plt.figure(figsize=(10,6))
sns.barplot(x='Algorithm', y='F1-score', hue='Sample', data=df)
plt.title('F1-score of Each Algorithm')
plt.legend(loc='lower right')
plt.ylabel('F1-score')
plt.xlabel('Algorithm')
plt.show()

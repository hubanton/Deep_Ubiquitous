import pandas as pd
from imblearn.over_sampling import SMOTE

from Dataloading import get_patient_data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2

X, y = get_patient_data(use_interpolation=False)

# y_counts = y.value_counts()
# plt.bar(['inactive', 'active'], y_counts, width=0.5, color=['royalblue', 'gold'])
# plt.show()

# n_components = 2
#
# pca_old_people = PCA(n_components=n_components)
# pca_transformed = pca_old_people.fit_transform(X)
#
# new_df = pd.DataFrame(data=pca_transformed, columns=[f'{i}' for i in range(n_components)])
#
# print(new_df.tail())
#
# print('Explained variation per principal component: {}'.format(pca_old_people.explained_variance_ratio_))


# labels = "Frontal axis", "Vertical axis", "Lateral axis", "Id", "RSSI", "Phase", "Frequency"
# sizes = pca_old_people.explained_variance_ratio_
#
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=180)
#
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.show()


# Select the top K most important features
k = 3
selector = SelectKBest(k=k)
X_train_selected = selector.fit_transform(X, y)

# Get the feature names
mask = selector.get_support()
print(mask)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.cluster import HDBSCAN
import umap

######################################
#--------BLOOD CELL CLUSTERING--------
######################################

# load data and visualise first few rows
data = pd.read_csv("C:/Users/c_reg/OneDrive/Desktop/SML/blood_cell_anomaly_detection.csv")
data.head()

# define features by dropping the label and other irrelevnt features
X = data.drop(['cell_id', 
               'cell_type', 
               'dataset_source', 
               'disease_category', 
               'staining_protocol', 
               'microscope_model',
               'patient_age_group',
               'patient_sex',
               'labeller_confidence_score'
               ], 
               axis=1)
# define label which is cell type
y = data["cell_type"]

# multiple features with different scales ruins accuracy
# we standardise the features to have mean 0 and variance 1
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

# split data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=730)

# storage for training and testing accuracies
train_acc = []
test_acc = []
# run kNN with different values of k
for k in range(1, 25):
    # define the classifier with k nearest neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # fit the model based on the training data
    knn.fit(X_train, y_train)
    # add the training and testing accuracies to the storage 
    train_acc.append(round(knn.score(X_train, y_train), 4))
    test_acc.append(round(knn.score(X_test, y_test), 4))
    # evaluate the model on the training and test data

# make sure k value are plotted frim k=1 not k=0
k_values = range(1, len(test_acc) + 1)
# plot test accuracy against k
plt.figure(figsize=(16, 6))
plt.plot(k_values, test_acc, label='Test Accuracy', marker='o',
         markersize=6, markerfacecolor='black', linestyle='-',
         color='black')
plt.plot(k_values, train_acc, label='Train Accuracy', marker='s',
         markersize=6, markerfacecolor='red', linestyle='--',
         color='red')
plt.xlabel('Number of Neighbours, K')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy versus K')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/knn_accuracy.png', dpi=300, bbox_inches='tight')

# select the best k and fit and predict 
knn_best = KNeighborsClassifier(n_neighbors=15)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

# form confusion matrix
cm = confusion_matrix(y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=knn_best.classes_, 
            yticklabels=knn_best.classes_)
plt.xlabel('Predicted Cell Type')
plt.ylabel('Actual Cell Type')
plt.title('Confusion Matrix (K=15)')
plt.savefig('figures/cell_confusion_matrix.png', dpi=300, bbox_inches='tight')


print(knn_best.score(X_test, y_test))
print(knn_best.score(X_train, y_train))

################################
#--------LOAN PREDICTION--------
################################

data_loan = pd.read_csv("C:/Users/c_reg/OneDrive/Desktop/SML/loan_risk_prediction_dataset.csv")
data_loan.head() 

# encode categorical variables into integerers so that they can be used in the model
data_loan_encoded = pd.get_dummies(data_loan, drop_first=True)

# define features and labels
X_loan = data_loan_encoded.drop('LoanApproved', axis=1)
y_loan = data_loan_encoded['LoanApproved']

X_train_loan, X_test_loan, y_train_loan, y_test_loan = train_test_split(
    X_loan, y_loan, test_size=0.2, random_state=472
)

rf = RandomForestClassifier(n_estimators=200,
                            max_depth=8,
                            oob_score=True, 
                            random_state=472)
rf.fit(X_train_loan, y_train_loan)

print('Train Accuracy:', round(rf.score(X_train_loan, y_train_loan), 4))
print('Test Accuracy:', round(rf.score(X_test_loan, y_test_loan), 5))

plt.figure(figsize=(20, 10))
plot_tree(
    rf.estimators_[6],  
    feature_names=X_loan.columns,
    class_names=y_loan.unique().astype(str),
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Single Decision Tree from Random Forest")
plt.savefig('figures/single_loan_decision_tree.png', dpi=300, bbox_inches='tight')



# Use the column names directly from your processed training data
feature_names = X_train_loan.columns 
importances = rf.feature_importances_

# Create a DataFrame for easy plotting
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')

plt.title('Feature Importance for Loan Classification')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/feature_importance_graph.png', dpi=300, bbox_inches='tight')

######################################
#--------GAIA STARS CLUSTERING--------
######################################


stars = pd.read_csv("C:/Users/c_reg/OneDrive/Desktop/SML/gaia-dr2-rave-35.csv")
stars.drop('source_id', axis=1, inplace=True)
stars.head()

# data is 250,000+ rows so sample 10000
stars_sample = stars.sample(100000, random_state=730).reset_index(drop=True)
# all different features have different scales so we standardise them to have mean 0 and variance 1
stars_scaled = scalar.fit_transform(stars_sample)
# define the umap dimension reduction technique to reduce the data to 2 dimensions for visualisation
reducer = umap.UMAP(n_components=2, n_jobs=1,random_state=730)
# perform umap dimension reduction on the scaled data
stars_umap = reducer.fit_transform(stars_scaled)

plt.figure(figsize=(12, 8))
plt.scatter(stars_umap[:, 0], stars_umap[:, 1],
            s=2, alpha=0.4, color='steelblue')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Projection')
plt.tight_layout()
plt.savefig('figures/umap_projection.png', dpi=300, bbox_inches='tight')

# define and fit the HDBSCAN clustering algorithm on the UMAP reduced data
hdb = HDBSCAN(min_cluster_size=300, 
              min_samples=10, 
              cluster_selection_method='eom'
              )
hdb.fit(stars_umap)
# get the cluster labels assigned by HDBSCAN
cluster_labels = hdb.labels_
# filter out noise points (label -1)
no_noise = cluster_labels != -1
stars_umap_filtered = stars_umap[no_noise]
cluster_labels_filtered = cluster_labels[no_noise]

plt.figure(figsize=(12, 8))
sns.scatterplot(x=stars_umap_filtered[:, 0], y=stars_umap_filtered[:, 1], 
                hue=cluster_labels_filtered, palette='tab10', s=50, alpha=0.7)
plt.title('HDBSCAN Clustering of Stars (Noise Removed)')  
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title='Cluster Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/HDBSCAN_UMAP_plot.png', dpi=300, bbox_inches='tight')

stars_sample['cluster'] = cluster_labels
print(stars_sample.groupby('cluster').mean().T)
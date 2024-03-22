import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from funzioni import *
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pgmpy.estimators import K2Score, HillClimbSearch
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination



#Caricamento dataset originale
dataset_path_original = '/Users/francescadimperio/Desktop/CONSEGNA_ICON/cervical-cancer_csv.csv'
dataset_original = pd.read_csv (dataset_path_original)
print("queste sono le colonne nel nostro dataset originale")
print(dataset_original.columns)
pd.set_option('display.max_columns', None)
print (dataset_original)

# Caricamento del dataset aggiornato
dataset_path = '/Users/francescadimperio/Desktop/CONSEGNA_ICON/cervical-cancer_aggiornato_csv.csv'
dataset = pd.read_csv(dataset_path)
print("queste sono le colonne nel nostro dataset aggiornato")
print(dataset.columns)
pd.set_option('display.max_columns', None)
print (dataset)

y = dataset['Dx:Cancer']

col_discrete = [col for col in dataset.columns if dataset[col].dtype == 'object']
col_continue = [col for col in dataset.columns if col not in col_discrete]

print('Numero di righe presenti nel Dataset:', dataset.shape[0])
print('Numero di colonne presenti nel Dataset:', dataset.shape[1])
print('\nLe colonne di tipologia discreta sono:\n', col_discrete)
print('\nLe colonne di tipologia continua sono:\n', col_continue)
print('\nValori nulli per ogni colonna:\n', dataset.isnull().sum())
print()

knn_imputer = KNNImputer(n_neighbors=5)
dataset_imputed = pd.DataFrame(knn_imputer.fit_transform(dataset), columns=dataset.columns)
valori_nulli_imputati = dataset_imputed.isnull().sum()
print(valori_nulli_imputati)

plt.figure(figsize=(6, 6))
health_counts = dataset['Dx:Cancer'].value_counts()
plt.pie(health_counts, labels=health_counts.index, autopct='%1.1f%%', startangle=90, colors=['purple', 'blue'])
plt.title('Distribuzione del cancro alla cervice')
plt.show()

visualizza_distribuzione_conteggio(dataset[['Dx:Cancer','Age']], 
                                  'Distribuzione casi di cancro alla cervice rispetto all\' età del paziente: ')

visualizza_distribuzione_conteggio(dataset[['Smokes','Dx:Cancer']], 
                                  'Distribuzione casi di cancro alla cervice se il paziente è fumatore: ')

visualizza_distribuzione_conteggio(dataset[['Dx:HPV','Dx:Cancer']], 
                                  'Distribuzione casi di cancro alla cervice rispetto alla diagnosi dell\' HPV : ')

visualizza_distribuzione_conteggio(dataset[['Schiller','Dx:Cancer']], 
                                  'Distribuzione casi di cancro alla cervice rispetto al test di Schiller : ')

visualizza_distribuzione_conteggio(dataset[['Biopsy','Dx:Cancer']], 
                                  'Distribuzione casi di cancro alla cervice rispetto alla biopsia : ')

scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

ds_stand = scaler_standard.fit_transform(dataset[col_continue])
ds_norm = scaler_minmax.fit_transform(dataset[col_continue])

#%%% APPRENDIMENTO NON SUPERVISIONATO

### CLUSTER PER SOTTOGRUPPI DEL CASO O

### DATASET ORIGINALE

X = dataset_imputed.drop(columns=['Dx:Cancer'])

fattori_selezionati = X.iloc[:, [2, 3, 4, 5, 9, 10, 12, 13]]  

Y = fattori_selezionati.values

num_clusters = 2
kmeans_original = KMeans(n_clusters=num_clusters, n_init=10)
kmeans_original.fit(Y)
cluster_centers = kmeans_original.cluster_centers_
labels = kmeans_original.labels_

distances = np.linalg.norm(Y - cluster_centers[labels], axis=1)
threshold = np.median(distances)

alto_rischio_indices = np.where(distances > threshold)[0]
basso_rischio_indices = np.where(distances <= threshold)[0]

num_alto_rischio = len(alto_rischio_indices)
num_basso_rischio = len(basso_rischio_indices)

cancro_alto_rischio = dataset_imputed['Dx:Cancer'][alto_rischio_indices].sum()
cancro_basso_rischio = dataset_imputed['Dx:Cancer'][basso_rischio_indices].sum()

labels = ['Alto rischio', 'Basso rischio', 'Cancro']
sizes = [num_alto_rischio, num_basso_rischio, cancro_alto_rischio + cancro_basso_rischio]
colors = ['purple', 'blue', 'pink']
explode = (0.1, 0, 0)  # Esplosione del primo cuneo (alto rischio)

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Garantisce che il grafico sia circolare
plt.title('Distribuzione dei casi di pazienti con cancro effettivo e ad alto e basso rischio dataset originale')
plt.show()


# VALORE DI SILHOUETTE E METODO DEL GOMITO

num_clusters_range = range(2, 16)

distortions = []

for num_clusters in num_clusters_range:
    kmeans_original = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans_original.fit(Y)
    distortions.append(kmeans_original.inertia_)

    silhouette_avg = silhouette_score(Y, kmeans_original.labels_)
    print('Con n_clusters={0}, il valore di silhouette è {1}'.format(num_clusters, silhouette_avg))

plt.figure(figsize=(8, 6))
plt.plot(num_clusters_range, distortions, marker='o')
plt.title('Metodo del gomito')
plt.xlabel('Numero di cluster')
plt.ylabel('Distorsione')
plt.xticks(num_clusters_range)
plt.grid(True)
plt.show()

### DATASET NORMALIZZATO

ds_norm_imputed = knn_imputer.fit_transform(ds_norm)
df_norm_imputed = pd.DataFrame(ds_norm_imputed, columns=dataset.columns)
X = df_norm_imputed.drop(columns=['Dx:Cancer'])

fattori_selezionati = X.iloc[:, [2, 3, 4, 5, 9, 10, 12, 13]]

Y = fattori_selezionati.values

num_clusters = 2
kmeans_norm = KMeans(n_clusters=num_clusters, n_init=10)
kmeans_norm.fit(Y)
cluster_centers = kmeans_norm.cluster_centers_
labels = kmeans_norm.labels_

distances = np.linalg.norm(Y - cluster_centers[labels], axis=1)
threshold = np.median(distances)

alto_rischio_indices = np.where(distances > threshold)[0]
basso_rischio_indices = np.where(distances <= threshold)[0]

num_alto_rischio = len(alto_rischio_indices)
num_basso_rischio = len(basso_rischio_indices)

cancro_alto_rischio = df_norm_imputed['Dx:Cancer'][alto_rischio_indices].sum()
cancro_basso_rischio = df_norm_imputed['Dx:Cancer'][basso_rischio_indices].sum()

labels = ['Alto rischio', 'Basso rischio', 'Cancro']
sizes = [num_alto_rischio, num_basso_rischio, cancro_alto_rischio + cancro_basso_rischio]
colors = ['purple', 'blue', 'pink']
explode = (0.1, 0, 0)  # Esplosione del primo cuneo (alto rischio)

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Garantisce che il grafico sia circolare
plt.title('Distribuzione dei casi di pazienti con cancro effettivo e ad alto e basso rischio con dataset normalizzato')
plt.show()

#VALORE DI SILHOUETTE E METODO DEL GOMITO

num_clusters_range = range(2, 16)

distortions = []

for num_clusters in num_clusters_range:
    kmeans_norm = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans_norm.fit(Y)
    distortions.append(kmeans_norm.inertia_)

    silhouette_avg = silhouette_score(Y, kmeans_norm.labels_)
   
    print('Con n_clusters={0}, il valore di silhouette è {1}'.format(num_clusters, silhouette_avg))

plt.figure(figsize=(8, 6))
plt.plot(num_clusters_range, distortions, marker='o')
plt.title('Metodo del gomito')
plt.xlabel('Numero di cluster')
plt.ylabel('Distorsione')
plt.xticks(num_clusters_range)
plt.grid(True)
plt.show()

### DATASET STANDARDIZZATO

ds_stand_imputed = knn_imputer.fit_transform(ds_stand)
df_stand_imputed = pd.DataFrame(ds_stand_imputed, columns=dataset.columns)
X = df_stand_imputed.drop(columns=['Dx:Cancer'])

fattori_selezionati = X.iloc[:, [2, 3, 4, 5, 9, 10, 12, 13]]

Y = fattori_selezionati.values

num_clusters = 2
kmeans_stand = KMeans(n_clusters=num_clusters, n_init=10)
kmeans_stand.fit(Y)
cluster_centers = kmeans_stand.cluster_centers_
labels = kmeans_stand.labels_

distances = np.linalg.norm(Y - cluster_centers[labels], axis=1)
threshold = np.median(distances)

alto_rischio_indices = np.where(distances > threshold)[0]
basso_rischio_indices = np.where(distances <= threshold)[0]

num_alto_rischio = len(alto_rischio_indices)
num_basso_rischio = len(basso_rischio_indices)

cancro_alto_rischio = df_stand_imputed['Dx:Cancer'][alto_rischio_indices].sum()
cancro_basso_rischio = df_stand_imputed['Dx:Cancer'][basso_rischio_indices].sum()

labels = ['Alto rischio', 'Basso rischio', 'Cancro']
sizes = [num_alto_rischio, num_basso_rischio, cancro_alto_rischio + cancro_basso_rischio]
colors = ['purple', 'blue', 'pink']
explode = (0.1, 0, 0)

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Distribuzione dei casi di pazienti con cancro effettivo e ad alto e basso rischio con dataset standardizzato')
plt.show()

#VALORE DI SILHOUETTE E METODO DEL GOMITO

num_clusters_range = range(2, 16)

distortions = []

for num_clusters in num_clusters_range:
    kmeans_stand = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans_stand.fit(Y)
    distortions.append(kmeans_stand.inertia_)

    silhouette_avg = silhouette_score(Y, kmeans_stand.labels_)
    
    print('Con n_clusters={0}, il valore di silhouette è {1}'.format(num_clusters, silhouette_avg))

plt.figure(figsize=(8, 6))
plt.plot(num_clusters_range, distortions, marker='o')
plt.title('Metodo del gomito')
plt.xlabel('Numero di cluster')
plt.ylabel('Distorsione')
plt.xticks(num_clusters_range)
plt.grid(True)
plt.show()


#%%% APPRENDIMENTO SUPERVISIONATO

X = dataset.drop(columns=['Dx:Cancer'])
y = dataset['Dx:Cancer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=54)

dataset_norm_imputed = pd.DataFrame(knn_imputer.fit_transform(dataset), columns=dataset.columns)
valori_nulli_imputati = dataset_norm_imputed.isnull().sum()
print(valori_nulli_imputati)

X_train_imputed = pd.DataFrame(knn_imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(knn_imputer.transform(X_test), columns=X_test.columns)

### KNN

k_values = range(1, 21)
knn_classifier = KNeighborsClassifier(n_neighbors=10)
knn_classifier.fit(X_train_imputed, y_train)
knn_classifier.feature_names_in_ = X.columns

### CROSS-VALIDATION

k_values = range(1, 25)  

cross_val_scores = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_classifier, X_train_imputed, y_train, cv=5, scoring='accuracy')
    cross_val_scores.append(scores.mean())

plt.plot(k_values, cross_val_scores, marker='o')
plt.title('Cross-Validation')
plt.xlabel('Numero di vicini (k)')
plt.ylabel('Accuratezza Media')
plt.show()

### ACCURATEZZA K (6-25) CROSS VALIDATION
k_values = range(6, 26)
cross_val_scores = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_imputed, y_train)
    scores = cross_val_score(knn_classifier, X_train_imputed, y_train, cv=5, scoring='accuracy')
    cross_val_scores.append(scores.mean())

plt.plot(k_values, cross_val_scores, marker='o')
plt.title('Cross-Validation')
plt.xlabel('Numero di vicini (k)')
plt.ylabel('Accuratezza Media')
plt.show()

### VISUALIZZAZIONE DEGLI IPERPARAMETRI

param_grid = {
    'n_neighbors': [6, 7, 8, 22, 23, 24],
    'p': [1, 2]
}

knn_classifier = KNeighborsClassifier()
grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train_imputed, y_train)

best_params = grid_search.best_params_
print("Iperparametri ottimali:", best_params)

accuracy = grid_search.best_score_
print("Accuratezza sul set di addestramento:", accuracy)

### MATRICE DI CONFUSIONE CON SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

knn_classifier_resampled = KNeighborsClassifier(n_neighbors=6, p=1)
knn_classifier_resampled.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = knn_classifier_resampled.predict(X_test_imputed)
accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
conf_matrix_resampled = confusion_matrix(y_test, y_pred_resampled)
class_report_resampled = classification_report(y_test, y_pred_resampled)

print('\n\n')
print(f'Accuratezza con SMOTE: {accuracy_resampled:.4f}\n')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_resampled, annot=True, fmt='d', cmap='Purples', xticklabels=['Cancro No', 'Cancro Si'],
            yticklabels=['Cancro No', 'Cancro Si'])
plt.title('Matrice di Confusione per KNN con SMOTE')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.show()
report_resampled = classification_report(y_test, y_pred_resampled, target_names=['Cancro No', 'Cancro Si'])

print("Classification Report per K-Nearest Neighbors (k=6) con SMOTE:\n")
print(report_resampled)


### MATRICE DI CONFUSIONE SENZA SMOTE

knn_classifier = KNeighborsClassifier(n_neighbors=6, p=1)
knn_classifier.fit(X_train_imputed, y_train)

y_pred = knn_classifier.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(y_test, y_pred)

print('\n\n')
print(f'Accuratezza senza SMOTE: {accuracy:.4f}\n')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=['Cancro No', 'Cancro Si'],
            yticklabels=['Cancro No', 'Cancro Si'])
plt.title('Matrice di Confusione per KNN senza SMOTE')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.show()
report = classification_report(y_test, y_pred, target_names=['Cancro No', 'Cancro Si'])

print("Classification Report per K-Nearest Neighbors (k=6) senza SMOTE:\n")
print(report)

### ACCURACY TRAINING E TEST SET E CURVA DI OVERFITTING CON SMOTE

knn_classifier = KNeighborsClassifier(n_neighbors=6, p=1)
knn_classifier.fit(X_train_imputed, y_train)

y_train_pred = knn_classifier.predict(X_train_imputed)
accuracy_train = accuracy_score(y_train, y_train_pred)

accuracy_test = accuracy_score(y_test, y_pred_resampled)

labels = ['Training Set', 'Test Set']
accuracies = [accuracy_train, accuracy_test]

plt.bar(labels, accuracies, color=['#FF00FF', 'purple'])
plt.ylabel('Accuratezza')
plt.title('Accuracy su Training Set e Test Set con SMOTE')
plt.ylim(0, 1) 
plt.show()

train_sizes_resampled, train_scores_resampled, test_scores_resampled = learning_curve(knn_classifier_resampled, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean_resampled = np.mean(train_scores_resampled, axis=1)
train_std_resampled = np.std(train_scores_resampled, axis=1)
test_mean_resampled = np.mean(test_scores_resampled, axis=1)
test_std_resampled = np.std(test_scores_resampled, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_resampled, train_mean_resampled, marker='o', label='Training Score', color='#FF00FF')
plt.fill_between(train_sizes_resampled, train_mean_resampled - train_std_resampled, train_mean_resampled + train_std_resampled, alpha=0.15, color='#FF00FF')
plt.plot(train_sizes_resampled, test_mean_resampled, marker='o', label='Test Score', color='purple')
plt.fill_between(train_sizes_resampled, test_mean_resampled - test_std_resampled, test_mean_resampled + test_std_resampled, alpha=0.15, color='purple')
plt.title('Curva di Overfitting per il KNN (k=6) senza SMOTE')
plt.xlabel('Dimensione del Set di Addestramento')
plt.ylabel('Accuratezza')
plt.legend(loc='best')
plt.grid(True)
plt.show()

print("Dimensioni del set di addestramento resampled:", X_train_resampled.shape)
print("Dimensioni del set di test:", X_test_imputed.shape)

### ACCURACY TRAINING E TEST SET E CURVA DI OVERFITTING SENZA SMOTE

y_train_pred = knn_classifier.predict(X_train_imputed)
accuracy_train = accuracy_score(y_train, y_train_pred)

accuracy_test = accuracy_score(y_test, y_pred)

labels = ['Training Set', 'Test Set']
accuracies = [accuracy_train, accuracy_test]

plt.bar(labels, accuracies, color=['#FF00FF', 'purple'])
plt.ylabel('Accuratezza')
plt.title('Accuracy su Training Set e Test Set senza SMOTE')
plt.ylim(0, 1) 
plt.show()

knn_classifier = KNeighborsClassifier(n_neighbors=6, p=1)

train_sizes, train_scores, test_scores = learning_curve(knn_classifier, X_train_imputed, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
print("Dimensione del set di addestramento: ", train_sizes)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, marker='o', label='Training Score', color='#FF00FF')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#FF00FF')

plt.plot(train_sizes, test_mean, marker='o', label='Test Score', color='purple')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='purple')

plt.title('Curva di Overfitting per KNN(k=6) con SMOTE')
plt.xlabel('Dimensione del Set di Addestramento')
plt.ylabel('Accuratezza')
plt.legend(loc='best')
plt.grid(True)
plt.show()

print("Dimensioni del set di addestramento:", X_train.shape)

print("Dimensioni del set di test:", X_test.shape)


#%%%REGRESSIONE LOGISTICA 

### MATRICE DI CONFUSIONE CON SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

log_reg_resampled = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)

log_reg_resampled.fit(X_train_resampled, y_train_resampled)

y_pred_resampled = log_reg_resampled.predict(X_test_imputed)

precision_resampled = precision_score(y_test, y_pred_resampled)
recall_resampled = recall_score(y_test, y_pred_resampled)
f1_resampled = f1_score(y_test, y_pred_resampled)
accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
conf_matrix_resampled = confusion_matrix(y_test, y_pred_resampled)

print('\n\n')
print(f'Accuratezza con SMOTE: {accuracy_resampled:.4f}\n')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_resampled, annot=True, fmt='d', cmap='Purples', xticklabels=['Cancro No', 'Cancro Si'],
            yticklabels=['Cancro No', 'Cancro Si'])
plt.title('Matrice di Confusione per la Regressione Logistica con SMOTE')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.show()

report_resampled = classification_report(y_test, y_pred_resampled, target_names=['Cancro No', 'Cancro Si'])

print("Classification Report per la Regressione Logistica con SMOTE:\n")
print(report_resampled)

### MATRICE DI CONFUSIONE SENZA SMOTE

log_reg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
log_reg.fit(X_train_imputed, y_train)
y_pred = log_reg.predict(X_test_imputed)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print('\n\n')
print(f'Accuratezza senza SMOTE: {accuracy:.4f}\n')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=['Cancro No', 'Cancro Si'],
            yticklabels=['Cancro No', 'Cancro Si'])
plt.title('Matrice di Confusione per la Regressione Logistica senza SMOTE')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.show()

report = classification_report(y_test, y_pred, target_names=['Cancro No', 'Cancro Si'])

print("Classification Report per la Regressione Logistica senza SMOTE:\n")
print(report)

### TEST AND TRAINING CON SMOTE

y_train_pred_resampled = log_reg_resampled.predict(X_train_resampled)
y_pred_resampled = log_reg_resampled.predict(X_test_imputed)
accuracy_train_resampled = accuracy_score(y_train_resampled, y_train_pred_resampled)
accuracy_test_resampled = accuracy_score(y_test, y_pred_resampled)

labels = ['Training Set', 'Test Set']
accuracies = [accuracy_train_resampled, accuracy_test_resampled]

plt.bar(labels, accuracies, color=['#FF00FF', 'purple'])
plt.ylabel('Accuratezza')
plt.title('Accuracy su Training Set e Test Set con SMOTE')
plt.ylim(0, 1) 
plt.show()

print("Dimensioni del set di addestramento:", X_train_resampled.shape)
print("Dimensioni del set di test:", X_test_imputed.shape)

train_sizes, train_scores, test_scores = learning_curve(log_reg_resampled, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, marker='o', label='Training Score', color='#FF00FF')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#FF00FF')
plt.plot(train_sizes, test_mean, marker='o', label='Test Score', color='purple')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='purple')
plt.title('Curva di Overfitting per la Regressione Logistica con SMOTE')
plt.xlabel('Dimensione del Set di Addestramento')
plt.ylabel('Accuratezza')
plt.legend(loc='best')
plt.grid(True)
plt.show()

### TEST AND TRAINING SENZA SMOTE

y_train_pred = log_reg.predict(X_train_imputed)
y_pred = log_reg.predict(X_test_imputed)
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_pred)

labels = ['Training Set', 'Test Set']
accuracies = [accuracy_train, accuracy_test]

plt.bar(labels, accuracies, color=['#FF00FF', 'purple'])
plt.ylabel('Accuratezza')
plt.title('Accuracy su Training Set e Test Set senza SMOTE')
plt.ylim(0, 1) 
plt.show()

print("Dimensioni del set di addestramento:", X_train_imputed.shape)
print("Dimensioni del set di test:", X_test_imputed.shape)

train_sizes, train_scores, test_scores = learning_curve(log_reg, X_train_imputed, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, marker='o', label='Training Score', color='#FF00FF')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#FF00FF')
plt.plot(train_sizes, test_mean, marker='o', label='Test Score', color='purple')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='purple')
plt.title('Curva di Overfitting per la Regressione Logistica senza SMOTE')
plt.xlabel('Dimensione del Set di Addestramento')
plt.ylabel('Accuratezza')
plt.legend(loc='best')
plt.grid(True)
plt.show()


### RANDOM FOREST

### MATRICE DI CONFUSIONE CON SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

random_forest_resampled = RandomForestClassifier(random_state=42)
random_forest_resampled.fit(X_train_resampled, y_train_resampled)
y_pred_rf_resampled = random_forest_resampled.predict(X_test_imputed)

precision_rf_resampled = precision_score(y_test, y_pred_rf_resampled)
recall_rf_resampled = recall_score(y_test, y_pred_rf_resampled)
f1_rf_resampled = f1_score(y_test, y_pred_rf_resampled)
accuracy_rf_resampled = accuracy_score(y_test, y_pred_rf_resampled)

conf_matrix_rf_resampled = confusion_matrix(y_test, y_pred_rf_resampled)

print('\n\n')
print(f'Accuratezza con SMOTE: {accuracy_rf_resampled:.4f}\n')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf_resampled, annot=True, fmt='d', cmap='Purples', xticklabels=['Benigno', 'Maligno'],
            yticklabels=['Benigno', 'Maligno'])
plt.title('Matrice di Confusione per il Random Forest con SMOTE')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.show()

report_rf_resampled = classification_report(y_test, y_pred_rf_resampled, target_names=['Benigno', 'Maligno'])

print("Classification Report per il Random Forest con SMOTE:\n")
print(report_rf_resampled)

### MATRICE DI CONFUSIONE SENZA SMOTE

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train_imputed, y_train)

y_pred_rf = random_forest.predict(X_test_imputed)

precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print('\n\n')
print(f'Accuratezza senza SMOTE: {accuracy_rf:.4f}\n')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Purples', xticklabels=['Benigno', 'Maligno'],
            yticklabels=['Benigno', 'Maligno'])
plt.title('Matrice di Confusione per il Random Forest senza SMOTE')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.show()

report_rf = classification_report(y_test, y_pred_rf, target_names=['Benigno', 'Maligno'])
print("Classification Report per il Random Forest senza SMOTE:\n")
print(report_rf)

### ACCURACY TRAINING E TEST SET E CURVA DI OVERFITTING CON SMOTE

accuracy_train_rf_resampled = accuracy_score(y_train_resampled, random_forest_resampled.predict(X_train_resampled))
accuracy_test_rf_resampled = accuracy_score(y_test, y_pred_rf_resampled)

labels = ['Training Set', 'Test Set']
accuracies = [accuracy_train_rf_resampled, accuracy_test_rf_resampled]

plt.bar(labels, accuracies, color=['#FF00FF', 'purple'])
plt.ylabel('Accuratezza')
plt.title('Accuracy su Training Set e Test Set con SMOTE')
plt.ylim(0, 1) 
plt.show()

print("Dimensioni del set di addestramento:", X_train_resampled.shape)
print("Dimensioni del set di test:", X_test_imputed.shape)

train_sizes_rf, train_scores_rf, test_scores_rf = learning_curve(random_forest_resampled, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean_rf = np.mean(train_scores_rf, axis=1)
train_std_rf = np.std(train_scores_rf, axis=1)
test_mean_rf = np.mean(test_scores_rf, axis=1)
test_std_rf = np.std(test_scores_rf, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_rf, train_mean_rf, marker='o', label='Training Score', color='#FF00FF')
plt.fill_between(train_sizes_rf, train_mean_rf - train_std_rf, train_mean_rf + train_std_rf, alpha=0.15, color='#FF00FF')
plt.plot(train_sizes_rf, test_mean_rf, marker='o', label='Test Score', color='purple')
plt.fill_between(train_sizes_rf, test_mean_rf - test_std_rf, test_mean_rf + test_std_rf, alpha=0.15, color='purple')
plt.title('Curva di Overfitting per il Random Forest con SMOTE')
plt.xlabel('Dimensione del Set di Addestramento')
plt.ylabel('Accuratezza')
plt.legend(loc='best')
plt.grid(True)
plt.show()

### ACCURACY TRAINING E TEST SET E CURVA DI OVERFITTING SENZA SMOTE

accuracy_train_rf = accuracy_score(y_train, random_forest.predict(X_train_imputed))
accuracy_test_rf = accuracy_score(y_test, y_pred_rf)

labels = ['Training Set', 'Test Set']
accuracies = [accuracy_train_rf, accuracy_test_rf]

plt.bar(labels, accuracies, color=['#FF00FF', 'purple'])
plt.ylabel('Accuratezza')
plt.title('Accuracy su Training Set e Test Set senza SMOTE')
plt.ylim(0, 1) 
plt.show()

train_sizes_rf, train_scores_rf, test_scores_rf = learning_curve(random_forest, X_train_imputed, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean_rf = np.mean(train_scores_rf, axis=1)
train_std_rf = np.std(train_scores_rf, axis=1)
test_mean_rf = np.mean(test_scores_rf, axis=1)
test_std_rf = np.std(test_scores_rf, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_rf, train_mean_rf, marker='o', label='Training Score', color='#FF00FF')
plt.fill_between(train_sizes_rf, train_mean_rf - train_std_rf, train_mean_rf + train_std_rf, alpha=0.15, color='#FF00FF')
plt.plot(train_sizes_rf, test_mean_rf, marker='o', label='Test Score', color='purple')
plt.fill_between(train_sizes_rf, test_mean_rf - test_std_rf, test_mean_rf + test_std_rf, alpha=0.15, color='purple')
plt.title('Curva di Overfitting per il Random Forest senza SMOTE')
plt.xlabel('Dimensione del Set di Addestramento')
plt.ylabel('Accuratezza')
plt.legend(loc='best')
plt.grid(True)
plt.show()

#%%% RETE BAYESIANA

dataset_rete = pd.read_csv (dataset_path)
 
dataset_rete_imputed = pd.DataFrame(knn_imputer.fit_transform(dataset_rete), columns=dataset_rete.columns)

for col in dataset_rete_imputed.columns:
    dataset_rete_imputed[col] = dataset_rete_imputed[col].astype(int)

df_RBayes = dataset_rete_imputed.copy()

print(df_RBayes.isnull().sum())
print(df_RBayes.dtypes)

max_parents = 2
hc_k2_simplified = HillClimbSearch(df_RBayes)
modello_k2_simplified = hc_k2_simplified.estimate(scoring_method=K2Score(df_RBayes), max_indegree=max_parents, max_iter=1000)

rete_bayesiana = BayesianNetwork(modello_k2_simplified.edges())
rete_bayesiana.fit(df_RBayes)

print("Nodi della rete bayesiana:")
for node in rete_bayesiana.nodes():
    print(node)

print("\nArchi nella rete bayesiana:")
for edge in rete_bayesiana.edges():
    print(edge)

def visualizza_rete_bayesiana(nodi, archi):
    grafo = nx.DiGraph()
    grafo.add_nodes_from(nodi)
    grafo.add_edges_from(archi)

    plt.figure(figsize=(10, 8))
    pos = nx.nx_agraph.graphviz_layout(grafo, prog='dot')
    node_colors = ['red' if node == nodi[0] else 'lightblue' for node in nodi]    
    nx.draw_networkx(grafo, pos, node_color='lightblue', node_size=500, alpha=0.8, arrows=True, arrowstyle='->', arrowsize=10, font_size=10, font_family='sans-serif')

    plt.title("Rete Bayesiana")
    plt.axis('off')
    plt.show()

nodi = ['Age', 'STDs: Time since first diagnosis', 'First sexual intercourse', 'Smokes (years)', 'Num of pregnancies', 'STDs: Time since last diagnosis',
        'Smokes', 'Number of sexual partners', 'STDs (number)', 'Smokes (packs/year)', 'STDs', 'STDs:HPV', 'STDs:vulvo-perineal condylomatosis', 'STDs:HIV',
        'STDs:condylomatosis', 'Biopsy', 'Dx:Cancer', 'STDs: Number of diagnosis', 'Dx:HPV', 'Dx', 'Citology', 'Dx:CIN', 'Schiller', 'Hinselmann']

archi= [('Age', 'STDs: Time since first diagnosis'), ('First sexual intercourse', 'Smokes (years)'), ('Smokes (years)', 'Age'), ('Smokes (years)', 'Smokes'),
        ('Smokes (years)', 'STDs (number)'), ('Smokes (years)', 'STDs: Time since last diagnosis'), ('Smokes (years)', 'Number of sexual partners'), 
        ('Num of pregnancies', 'STDs: Time since last diagnosis'), ('STDs: Time since last diagnosis', 'STDs: Time since first diagnosis'), 
        ('STDs: Time since last diagnosis', 'Age'), ('Smokes', 'Number of sexual partners'), ('STDs (number)', 'STDs'), 
        ('STDs (number)', 'STDs:vulvo-perineal condylomatosis'), ('STDs (number)', 'STDs:HIV'), ('Smokes (packs/year)', 'Smokes (years)'), 
        ('STDs', 'STDs:HPV'), ('STDs:HPV', 'Dx:Cancer'), ('STDs:vulvo-perineal condylomatosis', 'STDs:condylomatosis'), ('STDs:HIV', 'Biopsy'), 
        ('STDs:condylomatosis', 'STDs:HIV'), ('Biopsy', 'Schiller'), ('Dx:Cancer', 'Dx:HPV'), ('Dx:Cancer', 'Dx'), ('Dx:Cancer', 'Citology'), 
        ('STDs: Number of diagnosis', 'STDs (number)'), ('Dx:HPV', 'Biopsy'), ('Dx', 'Dx:HPV'), ('Dx:CIN', 'Dx'), ('Dx:CIN', 'Schiller'), 
        ('Schiller', 'Hinselmann'), ('Schiller', 'Citology')]

visualizza_rete_bayesiana(nodi, archi)

modello_bayesiano = BayesianNetwork(archi)

for column in dataset.columns:
    if column != 'Dx:Cancer':
        modello_bayesiano.add_node(column)

bayes_estimator = BayesianEstimator(modello_bayesiano, dataset)

cpds = [bayes_estimator.estimate_cpd(variable) for variable in modello_bayesiano.nodes]

for cpd in cpds:
    modello_bayesiano.add_cpds(cpd)

inferenza = VariableElimination(modello_bayesiano)

for variable in modello_bayesiano.nodes:
    cpd = modello_bayesiano.get_cpds(variable)
    min_value = cpd.values.min()
    max_value = cpd.values.max()
    print(f"Valori limite per la variabile '{variable}':")
    print(f"Minimo: {min_value}")
    print(f"Massimo: {max_value}")
    print("\n")

cancro_si = inferenza.query(variables=['Dx:Cancer'], evidence={'Age':36,
                                                               'STDs: Time since first diagnosis':16, 
                                                               'First sexual intercourse':20, 
                                                               'Smokes (years)':0, 
                                                               'Num of pregnancies':2, 
                                                               'STDs: Time since last diagnosis':16,
                                                               'Smokes':0, 
                                                               'Number of sexual partners':3, 
                                                               'STDs (number)':1, 
                                                               'Smokes (packs/year)':0, 
                                                               'STDs':1, 
                                                               'STDs:HPV':1, 
                                                               'STDs:vulvo-perineal condylomatosis':0, 
                                                               'STDs:HIV':0,
                                                               'STDs:condylomatosis':0, 
                                                               'Biopsy':0, 
                                                               'STDs: Number of diagnosis':1, 
                                                               'Dx:HPV':1, 
                                                               'Dx':1, 
                                                               'Citology':0, 
                                                               'Dx:CIN':0, 
                                                               'Schiller':0, 
                                                               'Hinselmann':0})
print('\nProbabilità per una donna di avere il cancro alla cervice: ')
print(cancro_si, '\n')

cancro_no = inferenza.query(variables=['Dx:Cancer'], evidence={'Age':28,
                                                               'STDs: Time since first diagnosis':2, 
                                                               'First sexual intercourse':16, 
                                                               'Smokes (years)':12, 
                                                               'Num of pregnancies':3, 
                                                               'STDs: Time since last diagnosis':2,
                                                               'Smokes':1, 
                                                               'Number of sexual partners':3, 
                                                               'STDs (number)':1, 
                                                               'Smokes (packs/year)':6, 
                                                               'STDs':1, 
                                                               'STDs:HPV':0, 
                                                               'STDs:vulvo-perineal condylomatosis':0, 
                                                               'STDs:HIV':1,
                                                               'STDs:condylomatosis':0, 
                                                               'Biopsy':0, 
                                                               'STDs: Number of diagnosis':1, 
                                                               'Dx:HPV':0, 
                                                               'Dx':0, 
                                                               'Citology':0, 
                                                               'Dx:CIN':0, 
                                                               'Schiller':0, 
                                                               'Hinselmann':0})
print('\nProbabilità per una donna di non avere il cancro alla cervice: ')
print(cancro_no, '\n')

y_pred_si = [cancro_si.values[1] > cancro_si.values[0]]
y_pred_no = [cancro_no.values[1] > cancro_no.values[0]]

y_pred = y_pred_si + y_pred_no

y_true = [1] * len(y_pred_si) + [0] * len(y_pred_no)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division='warn')
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Valori della rete bayesiana")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

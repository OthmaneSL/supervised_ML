import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fonctions_plot import plot_learning_curve,calculate_metrics_race,afficher_importance_features, correlation_train, correlation_test, plot_permutation_importance,plot_results,plot_confusion,calculate_fairness_metrics



# Charger les données depuis les fichiers CSV
file = '/Users/othmanesl/Desktop/Apprentissage_suppervise/acsincome_ca_complete_modified.csv'

# Charger le fichier CSV en tant que DataFrame
df = pd.read_csv(file, sep=';')

X_all = df.iloc[:, :-1]  # Sélectionner toutes les colonnes sauf la dernière
Y_all = df.iloc[:, -1]   # Sélectionner la dernière colonne

X_all = X_all.reset_index(drop=True)
Y_all = Y_all.reset_index(drop=True)

X_all, Y_all = shuffle(X_all, Y_all, random_state=1)

# Standardiser les fonctionnalités 
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

num_samples = int(len(X_all)*0.1)
#num_samples = 10000
X, Y = X_all[:num_samples], Y_all[:num_samples]

gradient_model = GradientBoostingClassifier()

# Division de l'ensemble de données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

gradient_model.fit(X_train, Y_train)

# Évaluer le modèle avec la validation croisée
cv_scores1 = cross_val_score(gradient_model, X_train, Y_train, cv=5)

print(f"Score de cross validation : {cv_scores1.mean()}")

# Faire des prédictions sur les données de test
y_pred1 = gradient_model.predict(X_test)

plot_learning_curve(gradient_model, X_train, Y_train)

# Afficher les résultats
print(f"Accuracy Score pour gradient sur l'ensemble de test: {accuracy_score(Y_test, y_pred1)}")
print(f"Classification Report gradient sur l'ensemble de test:\n{classification_report(Y_test, y_pred1)}")
print("\n")

######GRid search 
param_grid_gradientboost = {'learning_rate': [0.01, 0.1, 1]}#'n_estimators': [50, 100, 200]}#, 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 7]}

grid_search = GridSearchCV(gradient_model, param_grid_gradientboost, cv=5, scoring='accuracy')
grid_search.fit(X_train, Y_train)

# Afficher les meilleurs hyperparamètres pour chaque modèle
print(f"Meilleurs hyperparamètres pour {gradient_model.__class__.__name__}: {grid_search.best_params_}")

# Évaluer le modèle avec la validation croisée
cv_scores2 = cross_val_score(grid_search.best_estimator_, X_train, Y_train, cv=5)
print(f"Score de cross validation : {cv_scores2.mean()}")
# Calculer l'écart type des scores de la validation croisée
cv_scores_std = cv_scores2.std()
print(f"Écart type des scores en validation croisée pour {gradient_model.__class__.__name__}: {cv_scores_std}")

y_pred2 = grid_search.predict(X_test)

# Afficher les résultats
print(f"Accuracy Score pour {gradient_model.__class__.__name__} sur l'ensemble de test: {accuracy_score(Y_test, y_pred2)}")
print(f"Classification Report pour {gradient_model.__class__.__name__} sur l'ensemble de test:\n{classification_report(Y_test, y_pred2)}")
print("\n")


################# ANALYSE DU MODELE #################################################
afficher_importance_features(gradient_model , X_all)

plot_permutation_importance(gradient_model , X_test, Y_test, X_all.columns)

plot_confusion(Y_test, y_pred1)

plot_confusion(Y_test, y_pred2)

plot_results(gradient_model, Y_test, y_pred1, y_pred2, cv_scores1, cv_scores2)

correlation_train(X_all, X_train,Y_train) 

correlation_test(X_all, X_test, y_pred2)

############################################################################



############ PREDICTION POUR LE NEVADA #########################
df_ne_feat = pd.read_csv('/Users/othmanesl/Desktop/Apprentissage_suppervise/acsincome_ne_allfeaturesTP2.csv')
df_ne_lab = pd.read_csv('/Users/othmanesl/Desktop/Apprentissage_suppervise/acsincome_ne_labelTP2.csv')

X_all_ne = df_ne_feat[['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']]
y_all_ne = df_ne_lab['PINCP']

X_all_ne, y_all_ne = shuffle(X_all_ne, y_all_ne, random_state=1)

num_samples = int(len(X_all) * 0.1)

X_ne, y_ne = X_all_ne[:num_samples], y_all_ne[:num_samples]

# Séparation en ensembles d'entraînement et de test
X_train_ne, X_test_ne, y_train_ne, y_test_ne = train_test_split(X_ne, y_ne, test_size=0.2, random_state=1)

y_pred_ne = grid_search.predict(X_test_ne)

plot_confusion(y_test_ne, y_pred_ne)
# Afficher les résultats
print(f"Accuracy Score pour {gradient_model.__class__.__name__} sur l'ensemble de test nevada: {accuracy_score(y_test_ne, y_pred_ne)}")
print(f"Classification Report pour {gradient_model.__class__.__name__} sur l'ensemble de test nevada :\n{classification_report(y_test_ne, y_pred_ne)}")
print("\n")
############ PREDICTION POUR LE COLORADO #########################
df_co_feat = pd.read_csv('/Users/othmanesl/Desktop/Apprentissage_suppervise/acsincome_co_allfeaturesTP2.csv')
df_co_lab = pd.read_csv('/Users/othmanesl/Desktop/Apprentissage_suppervise/acsincome_co_labelTP2.csv')

X_all_co = df_co_feat[['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']]
y_all_co = df_co_lab['PINCP']

X_all_co, y_all_co = shuffle(X_all_co, y_all_co, random_state=1)

num_samples = int(len(X_all_co) * 0.1)
X_co, y_co = X_all_co[:num_samples], y_all_co[:num_samples]

# Séparation en ensembles d'entraînement et de test
X_train_co, X_test_co, y_train_co, y_test_co = train_test_split(X_co, y_co, test_size=0.2, random_state=1)

y_pred_co = grid_search.predict(X_test_co)

plot_confusion(y_test_co, y_pred_co)
# Afficher les résultats
print(f"Accuracy Score pour {gradient_model.__class__.__name__} sur l'ensemble de test colorado: {accuracy_score(y_test_co, y_pred_co)}")
print(f"Classification Report pour {gradient_model.__class__.__name__} sur l'ensemble de test colorado :\n{classification_report(y_test_co, y_pred_co)}")
print("\n")
######################################################################################################################
sensitive_attribute_sex = X_test['SEX']

# Appel de la fonction pour évaluer l'équité par rapport à la feature 'SEX'
calculate_fairness_metrics(Y_test, y_pred2, sensitive_attribute_sex)

race_labels = X_test['RAC1P']

# Appel de la fonction pour évaluer l'équité par rapport à la feature 'RAC1P'
calculate_metrics_race(Y_test , y_pred2, race_labels)
######################################ANALYSE DE LA FEATURE SEX SUR LE DATASET SANS LA FEATURE ################################################

# Charger les données depuis les fichiers CSV
file2 = '/Users/othmanesl/Desktop/Apprentissage_suppervise/acsincome_ca_complete_withoutsex.csv'

# Charger le fichier CSV en tant que DataFrame
df = pd.read_csv(file2, sep=';')

# Enregistrer la colonne 'SEX' dans une variable
#sex_column = df['SEX']

X_all2 = df.iloc[:, :-1]  # Sélectionner toutes les colonnes sauf la dernière
Y_all2 = df.iloc[:, -1]   # Sélectionner la dernière colonne

X_all2 = X_all2.reset_index(drop=True)
Y_all2 = Y_all2.reset_index(drop=True)
X_all2, Y_all2 = shuffle(X_all2, Y_all2, random_state=1)

# only use the first N samples to limit training time num_samples = int(len(X_all)*0.1)
num_samples = int(len(X_all2)*0.1)
X2, Y2 = X_all2[:num_samples], Y_all2[:num_samples]

gradient_model2 = GradientBoostingClassifier()

# Division de l'ensemble de données en ensembles d'entraînement et de test
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.2, random_state=1)

grid_search2 = GridSearchCV(gradient_model2 , param_grid_gradientboost, cv=5, scoring='accuracy')
grid_search2.fit(X_train2, Y_train2)

y_pred3 = grid_search2.predict(X_test2)

sensitive_attribute_sex = X_test['SEX']

# Appel de la fonction pour évaluer l'équité par rapport à la feature 'SEX'
calculate_fairness_metrics(Y_test, y_pred3, sensitive_attribute_sex)

###################################### ANALYSE DE LA FEATURE RAC1P SUR LE DATASET SANS LA FEATURE ################################################

# Charger les données depuis les fichiers CSV
file2 = '/Users/othmanesl/Desktop/Apprentissage_suppervise/acsincome_ca_complete_withoutrace.csv'

# Charger le fichier CSV en tant que DataFrame
df = pd.read_csv(file2, sep=';')

X_all2 = df.iloc[:, :-1]  # Sélectionner toutes les colonnes sauf la dernière
Y_all2 = df.iloc[:, -1]   # Sélectionner la dernière colonne

X_all2 = X_all2.reset_index(drop=True)
Y_all2 = Y_all2.reset_index(drop=True)
X_all2, Y_all2 = shuffle(X_all2, Y_all2, random_state=1)

# only use the first N samples to limit training time num_samples = int(len(X_all)*0.1)
num_samples = int(len(X_all2)*1)
X2, Y2 = X_all2[:num_samples], Y_all2[:num_samples]

randomforest_model2 = RandomForestClassifier()

# Division de l'ensemble de données en ensembles d'entraînement et de test
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.2, random_state=1)

grid_search2 = GridSearchCV(gradient_model2 , param_grid_gradientboost , cv=5, scoring='accuracy')
grid_search2.fit(X_train2, Y_train2)

y_pred3 = grid_search2.predict(X_test2)

race_labels = X_test['RAC1P']

# Appel de la fonction pour évaluer l'équité par rapport à la feature 'RACE'
calculate_metrics_race(Y_test , y_pred3, race_labels)

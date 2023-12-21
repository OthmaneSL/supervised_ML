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
from scipy.stats import pearsonr
from sklearn.inspection import permutation_importance



def plot_learning_curve(model, X, Y, train_sizes=np.linspace(0.0001, 1.0, 10), cv=5, scoring = 'accuracy'):
   
    train_sizes, train_scores, val_scores = learning_curve(model, X, Y, cv=cv, train_sizes=train_sizes, scoring = scoring)

    plt.figure()
    plt.title(f"Learning Curve for {model.__class__.__name__}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

def correlation_train(X_all, X_train, Y_train):
    correlations = {}
    for feature in X_all.columns:
        correlation, _ = pearsonr(X_train[feature], Y_train)
        correlations[feature] = correlation

    # Create a DataFrame to display the correlations
    correlation_df = pd.DataFrame({'Feature': list(correlations.keys()), 'Correlation': list(correlations.values())})

    # Sort the DataFrame by the absolute value of correlation in descending order
    correlation_df = correlation_df.reindex(correlation_df['Correlation'].abs().sort_values(ascending=False).index)

    # Display the correlation DataFrame
    print("Correlations between features and label:")
    print(correlation_df)
    # Plot the correlations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Correlation', y='Feature', data=correlation_df, palette='viridis')
    plt.title('Correlations between Features and Label')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.show()

def correlation_train_svm(X_all, X_train, Y_train):
        correlations = {}
        
        for col_index in range(X_all.shape[1]):
            feature = X_all.columns[col_index]
            correlation, _ = pearsonr(X_train[:, col_index], Y_train)
            correlations[feature] = correlation
    
        correlation_df = pd.DataFrame({'Feature': list(correlations.keys()), 'Correlation': list(correlations.values())})
    
        # Sort the DataFrame by the absolute value of correlation in descending order
        correlation_df = correlation_df.reindex(correlation_df['Correlation'].abs().sort_values(ascending=False).index)
    
        # Display the correlation DataFrame
        print("Correlations between features and label:")
        print(correlation_df)
        
        # Plot the correlations
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Correlation', y='Feature', data=correlation_df, palette='viridis')
        plt.title('Correlations between Features and Label')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Feature')
        plt.show()
        
def correlation_test(X_all, X_test, y_pred):
    correlations = {}
    for feature in X_all.columns:
        correlation, _ = pearsonr(X_test[feature], y_pred)
        correlations[feature] = correlation

    # Create a DataFrame to display the correlations
    correlation_df = pd.DataFrame({'Feature': list(correlations.keys()), 'Correlation': list(correlations.values())})

    # Sort the DataFrame by the absolute value of correlation in descending order
    correlation_df = correlation_df.reindex(correlation_df['Correlation'].abs().sort_values(ascending=False).index)

    # Display the correlation DataFrame
    print("Correlations between features and predicted label:")
    print(correlation_df)
    # Plot the correlations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Correlation', y='Feature', data=correlation_df, palette='viridis')
    plt.title('Correlations between Features and predicted Label')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.show()
    
def correlation_test_svm(X_all, X_test, y_pred):
    correlations = {}
    
    for col_index in range(X_all.shape[1]):
        feature = X_all.columns[col_index]
        correlation, _ = pearsonr(X_test[:, col_index], y_pred)
        correlations[feature] = correlation

    # Create a DataFrame to display the correlations
    correlation_df = pd.DataFrame({'Feature': list(correlations.keys()), 'Correlation': list(correlations.values())})

    # Sort the DataFrame by the absolute value of correlation in descending order
    correlation_df = correlation_df.reindex(correlation_df['Correlation'].abs().sort_values(ascending=False).index)

    # Display the correlation DataFrame
    print("Correlations between features and predicted label:")
    print(correlation_df)
    
    # Plot the correlations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Correlation', y='Feature', data=correlation_df, palette='viridis')
    plt.title('Correlations between Features and Predicted Label')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.show()


        
def plot_results(model, Y_test, y_pred1, y_pred2, cv_scores1, cv_scores2):
    # Extraire les scores pour chaque métrique de y_pred1
    metrics_y_pred1 = classification_report(Y_test, y_pred1, output_dict=True)
    precision_score_y_pred1 = metrics_y_pred1['weighted avg']['precision']
    recall_score_y_pred1 = metrics_y_pred1['weighted avg']['recall']
    f1_score_y_pred1 = metrics_y_pred1['weighted avg']['f1-score']
    accuracy_score_y_pred1 = accuracy_score(Y_test, y_pred1)

    # Extraire les scores pour chaque métrique de y_pred2
    metrics_y_pred2 = classification_report(Y_test, y_pred2, output_dict=True)
    precision_score_y_pred2 = metrics_y_pred2['weighted avg']['precision']
    recall_score_y_pred2 = metrics_y_pred2['weighted avg']['recall']
    f1_score_y_pred2 = metrics_y_pred2['weighted avg']['f1-score']
    accuracy_score_y_pred2 = accuracy_score(Y_test, y_pred2)

    cv_scores1 = cv_scores1.mean()
    cv_scores2 = cv_scores2.mean()

    model_names_metrics_compare = [f"{model.__class__.__name__} (Avant)", f"{model.__class__.__name__} (Après)"]

    # Créer un DataFrame pour les scores de chaque métrique
    scores_metrics_compare_df = pd.DataFrame({
        'Modèle': model_names_metrics_compare * 4,
        'Métrique': ['Précision', 'Recall', 'F1 Score', 'Accuracy'] * 2,#,'Cross validation score'] * 2,
        'Score': [precision_score_y_pred1, recall_score_y_pred1, f1_score_y_pred1, accuracy_score_y_pred1,#cv_scores1,
                precision_score_y_pred2, recall_score_y_pred2, f1_score_y_pred2, accuracy_score_y_pred2],#cv_scores2],
        'Groupe': ['y_pred1'] * 4 + ['y_pred2'] * 4
    })

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Métrique', y='Score', hue='Groupe', data=scores_metrics_compare_df, palette='viridis')
    plt.ylim(0, 1)  
    plt.title('Comparaison des scores entre y_pred1 et y_pred2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  

    # Ajouter les valeurs sur chaque barre
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.show()


def plot_confusion(Y_test, Y_pred):
    cm = confusion_matrix(Y_test, Y_pred)
    
    # Calculer le taux de vrai positif 
    true_positive_rate = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    
    # Calculer le taux de faux négatif
    false_negative_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1])

    plt.figure(figsize=(8, 6))
    
    # Afficher la matrice de confusion
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    
    plt.text(0.5, -0.1, f'Taux de Vrai Positif (Sensibilité): {true_positive_rate:.2f}', ha='center', va='center', transform=plt.gca().transAxes, color='green', fontsize=10)
    plt.text(0.5, -0.15, f'Taux de Faux Négatif: {false_negative_rate:.2f}', ha='center', va='center', transform=plt.gca().transAxes, color='red', fontsize=10)

    plt.title("Confusion Matrix")
    plt.show()
    
def calculate_fairness_metrics(y_true, y_pred, sensitive_attribute):
    # Convertir les valeurs de 'SEX' en booléen pour faciliter le calcul
    #pour svm il faut decommenter la ligne en dessous car c'est les valeurs standardiser 
    #is_sensitive = sensitive_attribute > 0
    is_sensitive = sensitive_attribute == 1
 

    # Filtrer les prédictions et les vraies valeurs pour les hommes
    y_true_male = y_true[is_sensitive]
    y_pred_male = y_pred[is_sensitive]
  
    # Filtrer les prédictions et les vraies valeurs pour les femmes
    y_true_female = y_true[~is_sensitive]
    y_pred_female = y_pred[~is_sensitive]
    
  
    # Calculer les métriques d'équité pour les hommes
    confusion_matrix_male = confusion_matrix(y_true_male, y_pred_male)
    FP_male = confusion_matrix_male[0][1]
    TN_male = confusion_matrix_male[0][0]
    TP_male = confusion_matrix_male[1][1]
    FN_male = confusion_matrix_male[1][0]

    TP_male = TP_male / (TP_male + FN_male)
    FP_male = FP_male / (FP_male + TN_male)

    # Calculer les métriques d'équité pour les femmes
    confusion_matrix_female = confusion_matrix(y_true_female, y_pred_female)

    FP_female = confusion_matrix_female[0][1]
    TN_female = confusion_matrix_female[0][0]
    TP_female = confusion_matrix_female[1][1]
    FN_female = confusion_matrix_female[1][0]

   
    TP_female = TP_female / (TP_female + FN_female)
    FP_female  = FP_female / (FP_female + TN_female)

    
    
    # Afficher la matrice de confusion avec les taux de TP et FP pour les hommes
    plt.figure(figsize=(8, 6))
    plt.title('Matrice de confusion - Hommes')
    sns.heatmap(confusion_matrix_male, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.text(0.5, -0.1, f'Taux de Vrai Positif: {TP_male:.2f}', ha='center', va='center', transform=plt.gca().transAxes, color='green', fontsize=10)
    plt.text(0.5, -0.15, f'Taux de Faux Positif: {FP_male:.2f}', ha='center', va='center', transform=plt.gca().transAxes, color='red', fontsize=10)
    plt.show()

    # Afficher la matrice de confusion avec les taux de TP et FP pour les femmes
    plt.figure(figsize=(8, 6))
    plt.title('Matrice de confusion - Femmes')
    sns.heatmap(confusion_matrix_female, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.text(0.5, -0.1, f'Taux de Vrai Positif: {TP_female:.2f}', ha='center', va='center', transform=plt.gca().transAxes, color='green', fontsize=10)
    plt.text(0.5, -0.15, f'Taux de Faux Positif: {FP_female:.2f}', ha='center', va='center', transform=plt.gca().transAxes, color='red', fontsize=10)
    plt.show()

    # Affichage des résultats
    print("Matrice de confusion2 - Hommes :\n", confusion_matrix_male)
    print(f"Vrais posisitifs  - Hommes : {TP_male}")
    print(f"Taux de faux positifs - Hommes : {FP_male}")

    print("\n")

    print("Matrice de confusion - Femmes :\n", confusion_matrix_female)
    print(f"Vrais positif - Femmes : {TP_female}")
    print(f"Taux de faux positifs - Femmes : {FP_female}")
    return TP_male, FP_male, TP_female, FP_female


def plot_permutation_importance(model, X, y, feature_names):
    # Calculer l'importance des permutations
    result = permutation_importance(model, X, y, n_repeats=10, random_state=1)

    # Extraire les importances moyennes et les noms des fonctionnalités
    importances = result.importances_mean
    std = result.importances_std

    # Créer un DataFrame pour afficher les importances des fonctionnalités
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances, 'Std Dev': std})

    # Trier le DataFrame par ordre décroissant d'importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Afficher le DataFrame
    print("Importance des permutations des features :")
    print(importance_df)

    # Tracer un diagramme à barres pour visualiser l'importance des permutations des features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Importance des permutations des features')
    plt.show()
    

def calculate_metrics_race(y_true, y_pred, race_labels):
    unique_races = np.unique(race_labels)
    metrics = {}

    for race in unique_races:
        # Filtrer les données pour la race actuelle
        indices = np.where(race_labels == race)
        y_true_race = np.array(y_true)[indices]
        y_pred_race = np.array(y_pred)[indices]

        # Calcul de la matrice de confusion
        cm = confusion_matrix(y_true_race, y_pred_race).ravel()
        # Extension de la matrice de confusion si nécessaire
        cm = np.append(cm, [0] * (4 - len(cm)))  # Ajoute des zéros si moins de 4 valeurs
        tn, fp, fn, tp = cm

        # Calcul du taux de vrai positif et faux positif
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

        metrics[race] = {'TPR': tpr, 'FPR': fpr}

        # Affichage des taux pour chaque race
        print(f"Race {race}: TPR = {tpr:.2f}, FPR = {fpr:.2f}")

    return metrics


def afficher_importance_features(model, X_all):

    # Obtenir l'importance des features
    feature_importances = model.feature_importances_

    # Créer un DataFrame pour afficher les importances des features
    feature_importance_df = pd.DataFrame({'Feature': X_all.columns, 'Importance': feature_importances})

    # Trier le DataFrame par ordre décroissant d'importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Afficher le DataFrame
    print("Importance des features pour le modèle :")
    print(feature_importance_df)

    # Tracer un diagramme à barres pour visualiser l'importance des features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Importance des features pour le modèle')
    plt.show()


# Student Dropout Prediction
[Voir en français](#prévision-du-décrochage-des-étudiants)

## Project Overview
The main goal of this project is to explore educational data, analyze features contributing to student dropout, and experiment with prediction models.

## Directory Structure
- `01_model_training.ipynb`: Performs exploratory data analysis (EDA) and trains the model.
- `02_model_registration.ipynb`: Registers the trained model in Azure ML's model registry.
- `best_xgb_pipeline.pkl`: The trained model saved as a pickle file, registered on Azure ML.
- `heatmap.png`: A heatmap showing feature correlation, with pairs having at least 0.50 correlation.
- `student_dropout_data.csv`: A dataset from Kaggle, provided by Portuguese researchers, containing data about university students.

## Installation
This project was developed using Python 3.10.14 and the following libraries:
- `xgboost`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `shap`

## Usage Instructions
Create an `env.yaml` file with the following keys under the `azure` section:
- `subscription_id`
- `resource_group`
- `workspace_name`

This is necessary to instantiate an `MLClient` object when registering the model in Azure ML.

## Results
The project is still ongoing, but some key findings so far include:
- Features like the number of curricular units approved, especially in the 2nd semester, are key predictors.
- Whether tuition fees are up-to-date also impacts student dropout.

Performance metrics using different models:
- **Multi-Class Logistic Regression**:
  - F1-score for "dropout": 0.78
  - F1-score for "enrolled": 0.39
  - F1-score for "graduate": 0.85
- **XGBoost**:
  - F1-score for "dropout": 0.80
  - F1-score for "enrolled": 0.49
  - F1-score for "graduate": 0.86

This project is ideal for those learning about model training with scikit-learn pipelines, as well as exploring educational data.

---

## Prévision du décrochage des étudiants

### Aperçu du projet
L'objectif principal de ce projet est d'explorer des données éducatives, d'analyser les caractéristiques contribuant au décrochage des étudiants et d'expérimenter avec des modèles de prédiction.

### Structure du répertoire
- `01_model_training.ipynb`: Effectue l'analyse exploratoire des données (EDA) et entraîne le modèle.
- `02_model_registration.ipynb`: Enregistre le modèle entraîné dans le registre de modèles Azure ML.
- `best_xgb_pipeline.pkl`: Le modèle entraîné sauvegardé sous forme de fichier pickle, enregistré sur Azure ML.
- `heatmap.png`: Une carte thermique montrant la corrélation des caractéristiques, avec des paires ayant au moins 0,50 de corrélation.
- `student_dropout_data.csv`: Un ensemble de données de Kaggle, fourni par des chercheurs portugais, contenant des informations sur les étudiants universitaires.

### Installation
Ce projet a été développé en Python 3.10.14 et utilise les bibliothèques suivantes :
- `xgboost`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `shap`

### Instructions d'utilisation
Créez un fichier `env.yaml` avec les clés suivantes sous la section `azure` :
- `subscription_id`
- `resource_group`
- `workspace_name`

Cela est nécessaire pour instancier un objet `MLClient` lors de l'enregistrement du modèle dans Azure ML.

### Résultats
Le projet est toujours en cours, mais certains résultats préliminaires montrent :
- Des caractéristiques telles que le nombre d'unités curriculaires approuvées, notamment au 2e semestre, sont des prédicteurs importants.
- Le fait que les frais de scolarité soient à jour influence également le décrochage des étudiants.

Mesures de performance en utilisant différents modèles :
- **Régression Logistique Multi-Classe** :
  - F1-score pour "décroché" : 0.78
  - F1-score pour "inscrit" : 0.39
  - F1-score pour "diplômé" : 0.85
- **XGBoost** :
  - F1-score pour "décroché" : 0.80
  - F1-score pour "inscrit" : 0.49
  - F1-score pour "diplômé" : 0.86

Ce projet est idéal pour ceux qui souhaitent apprendre l'entraînement de modèles avec les pipelines scikit-learn et explorer des données dans le domaine de l'éducation.

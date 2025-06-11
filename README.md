# Customer churn prediction
In this project you will see my analysis on customer churn of the Telco company.

# Objectif de l'étude 

Une companie de téléphonie fait face à un problème de désabonnnement de ses clients. Elle nous demande de :

>- Classer ses clients selon leurs probabilités de désabonnement
>- Identifier les clients qu'elle doit contacter et la rémise adaptée qu'elle doit proposer

# Plan d'analyse 

1-) **Inspection de la base de données :**
dimension, noms des variables, présence si valeurs manquantes, 
recodage des variables catégorielles

2-) **Analyses statistiques :**
* Analyse univariée : Représentations graphiques des variables, identification des types de distribution pour les variables quantitatives
* Analyse bivariées : test de khi-2 pour les variables qualitatives,test de corrélation et etc...

3-) **Machine Learning:**
* Division de la base de données
* entrainement et test du modèle
* Evaluation du modèle : Métrique choisie : AUC Score 
* Optimisation : gridSearcCV
* Prédiction sur le jeux de données

4-) **Réponses aux problématiques**
* Classement des clients selon leurs probabilités de désabonnement
* Proposition des discounts
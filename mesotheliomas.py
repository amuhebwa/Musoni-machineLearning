# -*- coding: utf-8 -*-
"""
Created on Tue May 03 18:30:40 2016

@author: AGGREY
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_name = "Mesothelioma/mesothelioma.csv"
df = pd.read_csv(file_name,low_memory=False)

dataset = df.drop(['city'], axis=1) # drop the column we don't want
dataset['gender'] = np.where(dataset['gender'] == 0, 'Female', 'Male')
dataset['dead or not'] = np.where(dataset['dead or not'] == 0, 'Dead', 'Alive')
dataset['class of diagnosis'] = np.where(dataset['class of diagnosis'] == 1, 'Healthy', 'Mesothelioma')

dataset.info()

#Distribution of age among patients
#dataset.age.hist(alpha=0.8)
#plt.title('Histogram showing age distribution among patients')
#plt.xlabel('Age')
#plt.ylabel('Count of Patients')
#plt.gcf().savefig('graphs/meso/age_distribution.png')


# class 0f diagonosis
#sns.kdeplot(df['class of diagnosis'], shade=True, color='salmon',alpha=.7)
#sns.rugplot(df['class of diagnosis'], alpha=1.0, color='salmon')
#plt.gcf().savefig('graphs/meso/class_of_diagnosis.png')
#sns.kdeplot(dataset['duration of asbestos exposure'], shade=True, color='dodgerblue',alpha=0.9)

#sns.FacetGrid(dataset, hue="class of diagnosis", size=5).map(sns.kdeplot, "duration of asbestos exposure").add_legend()

#sns.violinplot(x="class of diagnosis", y="duration of asbestos exposure", hue="gender", data=dataset,split=True, inner="stick", palette="Set1")
#plt.gcf().savefig('graphs/meso/violin1.png')
#sns.barplot(x="gender", y="age", hue="type of MM", data=dataset)
#plt.gcf().savefig('graphs/meso/gender_age.png')
#sns.countplot(x = "habit of cigarette", data=dataset, palette="Set1")
#plt.gcf().savefig('graphs/meso/cigerette.png')
#sns.countplot(x = "class of diagnosis", hue="dead or not", data=dataset, palette="winter");
#plt.gcf().savefig('graphs/meso/dead_or_not.png')

#sns.kdeplot(dataset['duration of asbestos exposure'], shade=True, color='dodgerblue',alpha=0.9)
#plt.gcf().savefig('graphs/meso/asbestos_exposure.png')
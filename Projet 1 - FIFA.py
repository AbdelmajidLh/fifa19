#!/usr/bin/env python
# coding: utf-8

# <div style="display: flex; text-align:center; background-color:  #4d5af3 ;" >
# <h1 style="margin: auto; padding: 30px; ">Projet 01 - Analyser les données FIFA 19<br>
# </h1>
# </div>
# 

# #####  Auteur : Abdelmajid EL HOU - Data Analyst - abdelmajid.elhou@gmail.com

# <div style="border-style: double;border-width: 5px;border-color: RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">Ce programme est divisé en 4 parties.</h3>
#     <div style="margin: auto;padding-left: 40px;padding-bottom: 20px;">
#         <lu>
#             <li>Partie 1 : Le projet </li>
#             <li>Partie 2 : Librairies et fonctions</li>
#             <li>Partie 3 : Exploration & visualisation des données</li>
#             <li>Partie 4 : Visualisation avancée des donnéess</li>
#         </lu>
#     </div>
# </div>

# <div style="background-color: RGB(51,165,182);" >
# <h2 style="margin: auto; padding: 20px; color:#fff; ">Partie 1 - Le projet</h2>
# </div>

# FIFA 19 is a football simulation video game developed by EA Vancouver as part of Electronic Arts' FIFA series. Announced on 6 June 2018 for its E3 2018 press conference, it was released on 28 September 2018 for PlayStation 3, PlayStation 4, Xbox 360, Xbox One, Nintendo Switch, and Microsoft Windows.It is the 26th installment in the FIFA series. As with FIFA 18, Cristiano Ronaldo initially as the cover athlete of the regular edition: however, following his unanticipated transfer from Spanish club Real Madrid to Italian side Juventus, new cover art was released, featuring Neymar, Kevin De Bruyne and Paulo Dybala.
# 
# The game features the UEFA club competitions for the first time, including the UEFA Champions League and UEFA Europa League. Martin Tyler and Alan Smith return as regular commentators, while the new commentary team of Derek Rae and Lee Dixon feature in the UEFA competitions mode.Composer Hans Zimmer and rapper Vince Staples recorded a new remix of the UEFA Champions League anthem specifically for the game. The character Alex Hunter, who first appeared in FIFA 17 returns for the third and final installment of "The Journey", entitled, "The Journey: Champions".In June 2019, a free update added the FIFA Women's World Cup as a separate game mode.
# 
# This is the last game in the FIFA series to be available on a seventh-generation console, and the last known game to be available for the PlayStation 3 worldwide.
# 
# Inspired from: https://www.kaggle.com/roshansharma/fifa-data-visualization
# 
# <p><b>The idea of this project is to explor the FIFA dataset and doing some visualisations</b></p>

# <div style="background-color: RGB(51,165,182);" >
# <h2 style="margin: auto; padding: 20px; color:#fff; ">Partie 2 - Librairies et fonctions</h2>
# </div>

# In[158]:


# importer les librairies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[159]:


# Créer le dossier RES (résultats) s'il n'existe pas
import os

path = 'RES'

# Check whether the specified path exists or not
isExist = os.path.exists(path)

if not isExist:
  
  # Create a new directory (RES) if not exist 
  os.makedirs(path)
  print("The new directory is created!")


# <div style="background-color: RGB(51,165,182);" >
# <h2 style="margin: auto; padding: 20px; color:#fff; ">Partie 3 - Exploration & visualisation des données</h2>
# </div>

# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">3.1. - Importer & Nettoyer les données</h3>
# </div>

# In[160]:


# importer le fichier csv: https://www.kaggle.com/karangadiya/fifa19?select=data.csv
# https://www.kaggle.com/roshansharma/fifa-data-visualization
df =pd.read_csv('data/fifa.csv')


# In[161]:


# afficher l'entete du tableau
df.head()


# In[162]:


# Résumé des données
df.describe()


# In[163]:


# Vérifier les données manquantes
df.isnull().sum()


# In[164]:


df['FKAccuracy'].head()


# In[165]:


# Nettoyage des données: remplacer les valeurs maquantes par la moyenne de la population
df['ShortPassing'].fillna(df['ShortPassing'].mean(), inplace = True)
df['Volleys'].fillna(df['Volleys'].mean(), inplace = True)
df['Dribbling'].fillna(df['Dribbling'].mean(), inplace = True)
df['Curve'].fillna(df['Curve'].mean(), inplace = True)
df['FKAccuracy'].fillna(df['FKAccuracy'], inplace = True)
df['FKAccuracy'].fillna(df['FKAccuracy'], inplace = True)
df['LongPassing'].fillna(df['LongPassing'].mean(), inplace = True)
df['BallControl'].fillna(df['BallControl'].mean(), inplace = True)
df['HeadingAccuracy'].fillna(df['HeadingAccuracy'].mean(), inplace = True)
df['Finishing'].fillna(df['Finishing'].mean(), inplace = True)
df['Crossing'].fillna(df['Crossing'].mean(), inplace = True)
df['Weight'].fillna('200lbs', inplace = True)
df['Contract Valid Until'].fillna(2019, inplace = True)
df['Height'].fillna("5'11", inplace = True)
df['Loaned From'].fillna('None', inplace = True)
df['Joined'].fillna('Jul 1, 2018', inplace = True)
df['Jersey Number'].fillna(8, inplace = True)
df['Body Type'].fillna('Normal', inplace = True)
df['Position'].fillna('ST', inplace = True)
df['Club'].fillna('No Club', inplace = True)
df['Work Rate'].fillna('Medium/ Medium', inplace = True)
df['Skill Moves'].fillna(df['Skill Moves'].median(), inplace = True)
df['Weak Foot'].fillna(3, inplace = True)
df['Preferred Foot'].fillna('Right', inplace = True)
df['International Reputation'].fillna(1, inplace = True)
df['Wage'].fillna('€200K', inplace = True)


# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">3.2. - Nombre de joueurs par pays</h3>
# </div>

# In[166]:


# Nombre de joueurs par pays
df['Nationality'].value_counts().head()


# In[167]:


# Top 5 des pays en nombre de joueurs
top5 = list(df['Nationality'].value_counts().head().keys())

# nombre de joueurs pour chaque pays (top5)
top5_values = list(df['Nationality'].value_counts().head())

# Représentation graphique (sauvegarder les plots)
plt.figure(figsize=(9,5))
#plt.grid(b=None)
plt.bar(top5, top5_values)
plt.xlabel('Pays')
plt.ylabel('Nombre de joueurs')
plt.savefig('RES/top_5_pays_nombre_joueurs.png', dpi = 199) # save plots 
plt.show()


# L'England est le pays avec le plus de joueurs, suivi par l'Allemagne et l'espagne.

# In[168]:


# Les 5 pays avec le moin de joueurs
lowest5 = list(df['Nationality'].value_counts().tail().keys())

# nombre de joueurs pour chaque pays (top5)
lowest5_values = list(df['Nationality'].value_counts().tail())

# Représentation graphique (sauvegarder les plots)
plt.figure(figsize=(9,5))
#plt.grid(b=None)
plt.bar(lowest5, lowest5_values)
plt.xlabel('Pays')
plt.ylabel('Nombre de joueurs')
plt.savefig('RES/lowest_5_pays_nombre_joueurs.png', dpi = 199) # save plots 
plt.show()


# In[169]:


# Les différentes nationalités participant à la FIFA 2019

#plt.style.use('dark_background')
df['Nationality'].value_counts().head(80).plot.bar(color = 'orange', figsize = (20, 7))
plt.title('Les différentes nationalités participant à la FIFA 2019', fontsize = 30, fontweight = 20)
plt.xlabel('Paysy')
plt.ylabel('Nombre')
plt.show()


# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">3.3. - Nombre de buts par joueur</h3>
# </div>

# In[170]:


# Top 5 des joueurs et leurs Nationalité et club
df[['Name', 'Nationality', 'Club', 'Overall']].sort_values(by='Overall', ascending=False).head()


# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">3.4. - Salaire des joueur</h3>
# </div>

# In[171]:


# Adapter la colonne salaire 'Wage': remplacer les symboles euro et k
df.Wage = df.Wage.str.replace('€', '')
df.Wage = df.Wage.str.replace('K', '').astype('float')


# In[172]:


# Joueurs avec le salaire le plus haut (en milliers d'euro)
df[['Name', 'Wage']].sort_values(by='Wage', ascending=False).head()


# In[173]:


# Preffered foot
plt.figure(figsize=(8,6))
sns.countplot(df['Preferred Foot'])
plt.xlabel('Preferred Foot')
plt.ylabel('Nombre')
plt.savefig('RES/coté_preferé_des_joueurs.png', dpi = 199) # save plots 
plt.show()


# La majorité des joueurs prèfèrent jouer à droite. 

# La majorité des joueurs prèfèrent jouer la droite. 

# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">3.5. - Âge des joueur</h3>
# </div>

# In[174]:


# Histogramme: La distribution d'âge 
sns.set(style = "dark", palette = "colorblind", color_codes = True)
x = df.Age
plt.figure(figsize = (15,8))
ax = sns.distplot(x, bins = 58, kde = False, color = 'g')
ax.set_xlabel(xlabel = "Âge des joueurs", fontsize = 16)
ax.set_ylabel(ylabel = 'Nombre de joueurs', fontsize = 16)
ax.set_title(label = "Histogramme d'âge des joueurs", fontsize = 20)
plt.savefig('RES/histogram_age_joueurs.png', dpi = 199)
plt.show()


# In[175]:


# Liste des joueurs les plus jeunes
age_min = df['Age'].min()
df[(df.Age==age_min)][['Name', 'Age']]


# In[176]:


# Liste des joueurs les plus agés
age_max = df['Age'].max()
df[(df.Age==age_max)][['Name', 'Age']]


# O. Pérez agé de 45 ans est le joueur le plus agé.

# In[177]:


# Age vs rating
plt.figure(figsize=(18,6))
sns.barplot(x='Age', y='Overall', data=df)
plt.xlabel('Âge')
plt.ylabel('Overall')
plt.savefig('RES/overall_par_age.png', dpi = 199) # save plots 
plt.show()


# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">3.6. - Taille vs note (overall) des joueur</h3>
# </div>

# In[178]:


# Age vs rating
plt.figure(figsize=(18,6))
sns.barplot(x='Height', y='Overall', data=df)
plt.xlabel('Height')
plt.ylabel('Overall')
plt.savefig('RES/overall_par_Height.png', dpi = 199) # save plots 
plt.show()


# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">3.7. -  Meilleurs shot Power</h3>
# </div>

# In[179]:


# Les 5 meilleurs puissance de tir (ShotPower)
df[['Name', 'ShotPower']].sort_values(by='ShotPower', ascending=False).head()


# In[180]:


# Les 5 Long shot players
df[['Name', 'LongShots']].sort_values(by='LongShots', ascending=False).head()


# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">3.8. -  Meilleurs pénalties</h3>
# </div>

# In[181]:


# Top 5 pénalties (joueurs)
df[['Name', 'Penalties']].sort_values(by='Penalties', ascending=False).head()


# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">3.9. -  Nombre de joueurs sur la base de leurs mouvements de compétences.</h3>
# </div>

# In[182]:


plt.figure(figsize = (10, 8))
ax = sns.countplot(x = 'Skill Moves', data = df, palette = 'pastel')
#ax.set_title(label = ' Nombre de joueurs sur la base de leurs mouvements de compétences', fontsize = 20)
ax.set_xlabel(xlabel = 'Nombre de compétences', fontsize = 16)
ax.set_ylabel(ylabel = 'Nombre de joueurs', fontsize = 16)
plt.savefig('RES/mouvement_de_competences.png', dpi = 199)
plt.show()


# <div style="background-color: RGB(51,165,182);" >
# <h2 style="margin: auto; padding: 20px; color:#fff; ">Partie 4 - Visualisation avancée des données</h2>
# </div>

# In[183]:


# Choisir certaines variables importantes
important_var = ['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',
                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',
                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',
                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']

data_selected = pd.DataFrame(df, columns = important_var)


# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">4.1. -  Comparaison des scores globaux et de l'âge par rapport au pied préféré.</h3>
# </div>

# In[184]:


# Overall score et l'âge des joueurs - Faire un violin plot 

plt.rcParams['figure.figsize'] = (20, 7)
sns.boxenplot(df['Overall'], df['Age'], hue = df['Preferred Foot'], palette = 'Greys')
plt.title("Comparaison des scores globaux et de l'âge par rapport au pied préféré", fontsize = 20)
plt.savefig("RES/violin_plot.png", dpi=199)
plt.show()


# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 10px; color: RGB(51,165,182); ">4.2. -  Heatmap corrélations entre variables.</h3>
# </div>

# In[185]:


# Afficher les heatmap des corréations entre variables

plt.rcParams['figure.figsize'] = (30, 20)
sns.heatmap(data_selected[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',
                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',
                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',
                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']].corr(), annot = True)

plt.title('Heatmap de la dataset', fontsize = 30)
plt.savefig("RES/heatmap_variables.png", dpi=199)
plt.show()


# In[186]:


# Meilleurs joueurs par position avec leur âge, club et nationalité basé sur la score global (overall scores)

df.iloc[df.groupby(df['Position'])['Overall'].idxmax()][['Position', 'Name',
                                                               'Age', 'Club', 'Nationality']].style.background_gradient('Reds')


# In[187]:


data_pays.head()


# In[189]:


# supprimer le suffixe lbs
df.Weight = df.Weight.str.replace('lbs', '')

# choisir quelques pays
pays = ['Morocco', 'Germany', 'Sudan', 'France', 'Algeria', 'Brazil', 'Saudi Arabia', 'Russia']
data_pays = df.loc[df['Nationality'].isin(pays) & df['Weight']]

# convertir la colonne Weight en float 
data_pays["Weight"] = data_pays["Weight"].astype('float')

# violin plot
plt.rcParams['figure.figsize'] = (15, 7)
#ax = sns.violinplot(x=data_pays['Nationality'], y = data_pays['Weight'], palette = 'Reds')
ax = sns.violinplot(data=data_pays, x='Nationality', y='Weight')
ax.set_xlabel(xlabel = 'Pays', fontsize = 9)
ax.set_ylabel(ylabel = 'Poids en lbs', fontsize = 9)
ax.set_title(label = 'Distribution du poid des joueurs par pays', fontsize = 14)
plt.savefig('RES/poids_joueurs_pays.png', dpi = 199)
plt.show()


# On observe une variabilité différente du poids des joueurs selon la nationalité

# In[ ]:





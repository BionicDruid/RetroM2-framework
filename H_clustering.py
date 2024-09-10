import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

#Using z-score normalization
def normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data-mean)/std

#Function to graphs dendrograms given a dataset,showing the different linkage methods
#It also saves the plots as images
def graph_dendrogram(data, values, labels, name):
    # Hierarchical clustering using different linkage methods
    single_linkage = linkage(data, 'single')
    complete_linkage = linkage(data, 'complete')
    average_linkage = linkage(data, 'average')

    # Function to format labels with values
    def format_label(index):
        return f"{labels[index]} ({values[index]:.2f})"

    # Graphing the dendrogram with titles, comparing different linkage methods, and creating labels with values of the given feature, with title
    plt.figure(figsize=(10, 7))
    plt.title(f'{name} Single Linkage')
    dendrogram(single_linkage,
               orientation='top',
               labels=[format_label(i) for i in range(len(labels))],
               distance_sort='descending',
               show_leaf_counts=True)
    plt.xticks(rotation=90)
    plt.ylabel('Cluster Distance')
    plt.savefig(f'{name}_single.jpg', bbox_inches='tight', dpi=150)

    #Average linkage
    plt.figure(figsize=(10, 7))
    plt.title(f'{name} Average Linkage')
    dendrogram(average_linkage,
               orientation='top',
               labels=[format_label(i) for i in range(len(labels))],
               distance_sort='descending',
               show_leaf_counts=True)
    plt.xticks(rotation=90)
    plt.savefig(f'{name}_avg.jpg', bbox_inches='tight', dpi=150)

    #Complete linkage
    plt.figure(figsize=(10, 6))
    plt.title(f'{name} Complete Linkage')
    dendrogram(complete_linkage,
               orientation='top',
               labels=[format_label(i) for i in range(len(labels))],
               distance_sort='descending',
               show_leaf_counts=True)
    plt.xticks(rotation=90)
    plt.savefig(f'{name}_complete.jpg', bbox_inches='tight', dpi=150)

#Reading the data, we are using a very small dataset to make the analysis easier, only 30 movies, in comparison to the 250 of the original dataset
X = pd.read_csv('validation_dataset.csv')


#Keeping the original data and dividing into different features
rating = X['Rating']
votes = X['Votes']
revenue = X['Revenue (Millions)']
metascore = X['Metascore']

#Normalizing the data
normal_rating = normalization(rating)
normal_votes = normalization(votes)
normal_revenue = normalization(revenue)
normal_metascore = normalization(metascore)

# Getting the labels for the X axis
labels = X['Title'].values

# Graphing dendrograms for each feature
graph_dendrogram(normal_rating.values.reshape(-1, 1),rating, labels, "Rating")
graph_dendrogram(normal_votes.values.reshape(-1, 1),votes, labels, "Votes")
graph_dendrogram(normal_revenue.values.reshape(-1, 1),revenue, labels, "Revenue")
graph_dendrogram(normal_metascore.values.reshape(-1, 1),metascore, labels, "Metascore")

# Print dataset description
print(X.describe().transpose())

#Since this is an unsupervised learning problem, the analysis is going to be done by observing the dendrograms.
#In which we can see the different clusters that can be formed using the different linkage methods.
#For example the metascore using single linkage shows a better relationship with the quality of movies.
#By this I mean that we can see how the movies with the highest metascore remain grouped together, and the same for the lowest metascore.
#Then we can infer that the single linkage method is the best for the metascore portion of the dataset.

#When using the rating as a metric, we can see that the single linkage method separates the movies based on there will be blood.
#Since this movie has the highest rating, it is separated from the rest of the movies, while it is useful, we can learn more from the other linkage methods.
#When using average linkage, we can observe that we have small jumps in the quality of the movies and the separation in two groups which we can know that are the ones with higher or lower quality.
#But with complete linkage we can get the smallest jumps, keeping the qualityt of the movies more consistent.

#This are just a few of the examples of information we can obtain by reading and examining the dendrograms.

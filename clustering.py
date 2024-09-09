import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn.datasets import load_iris
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt


#Using z-score normalization
def normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data-mean)/std

#Function to graphs dendrograms given a dataset,showing the different linkage methods
#It also saves the plots as images
def graph_dendrogram(data):

    #Herarchical clustering using checking for each linkage method
    single_linkage = linkage(data, 'single')
    complete_linkage = linkage(data, 'complete')
    average_linkage = linkage(data, 'average')
   
    #Graphing the dendrogram with titles, comparing the different linkage methods
    labels = X['Title'].values
    plt.figure(figsize=(10, 7))
    plt.title('Single Linkage')
    dendrogram(single_linkage,
            orientation='top',
            labels=labels,
            distance_sort='descending',
            show_leaf_counts=True)
    plt.xticks(rotation=90)
    plt.ylabel('Cluster Distance')
    #Saving the plot as an image
    #plt.savefig('single.jpg', bbox_inches='tight', dpi=150)
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.title('Average Linkage')
    dendrogram(average_linkage,
            orientation='top',
            labels=labels,
            distance_sort='descending',
            show_leaf_counts=True)
    #Saving the plot as an image
    #plt.savefig('avg.jpg', bbox_inches='tight', dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title('Complete Linkage')
    dendrogram(complete_linkage,
            orientation='top',
            labels=labels,
            distance_sort='descending',
            show_leaf_counts=True)
    #Saving the plot as an image
    #plt.savefig('complete.jpg', bbox_inches='tight', dpi=150)
    plt.show()

#Reading the data
X = pd.read_csv('validation_dataset.csv')

#Dividing into different features
rating = X['Rating']
votes = X['Votes']
revenue = X['Revenue (Millions)']

#Normalizing the data
normal_rating = normalization(rating)
normal_votes = normalization(votes)
normal_revenue = normalization(revenue)

graph_dendrogram(normal_rating)
# graph_dendrogram(normal_votes)
# graph_dendrogram(normal_revenue)

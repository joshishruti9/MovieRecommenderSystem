# -*- coding: utf-8 -*-
"""
Created on Fri July 1 20:05:02 2019

@author: shruti
"""

import pandas as pd
import numpy as np

def kNearestneighboursVectorized(userIndex):
    userVector = movie_features_df.iloc[userIndex] # vector of input user Id
    lengthUserVector = (np.sum(np.square(userVector)))**0.5 # Euclidean Length of userVector
    # 2 dimensional array with rows representing users and columns representing movies
    matrix = np.array(movie_features_df)
    dotProduct = np.dot(matrix,userVector) # vectorized implementation. Computes numerators of all the cosine similarities
    # Process to compute denominators of all the cosine similarities  
    matrixSquare = np.square(matrix)
    matrixSquareRowWiseSum = np.sum(matrixSquare,axis=1)
    matrixLengths = np.sqrt(matrixSquareRowWiseSum)
    absMul = np.multiply(matrixLengths, lengthUserVector)
	# compute cosine similarities by dividing respective numerators by denominators
    arrF = np.divide(dotProduct,absMul) 
    return list(arrF) # return a list of cosine similarities between all the users and input user

def predictMovieRating(userId, movieId, k):
    cosineSimilaritiesList = kNearestneighboursVectorized(userId) # receives list of cosine similarities
    matrix = np.array(movie_features_df) # 2-dimensional numpy array of dataframe. row represents user.
    matrix = np.transpose(matrix) # transpose of a matrix to get movie as a row.
    # sorted list of cosine similarities in descending order
    sortedCosineSimilaritiesList = sorted(cosineSimilaritiesList,reverse=True) 
    #print(sortedCosineSImilaritiesList)
    minIndices = [] # list to store indices of k nearest users
    for i in range(1,k+1):
        val = sortedCosineSimilaritiesList[i] # get the value of cosine similarity from sorted list
        minIndex = cosineSimilaritiesList.index(val) # get position/index of that value
        minIndices.append(minIndex) # append that index to the list
    finalUserVector = sortedCosineSimilaritiesList[1:k+1] # k maximum cosine similarities
    #print(minIndices)
    #print(finalUserVector)
    
    movieVector = matrix[movieId] # fetches input movie row from transposed matrix
    finalMovievector = [] # list to contain ratings of k nearest users for input movie
    for i in range(len(minIndices)):
        index = minIndices[i] # retrieve user Id
        finalMovievector.append(movieVector[index]) # append rating by that user to the list
    arr1 = np.array(finalUserVector) 
    arr2 = np.array(finalMovievector)
    numeratorWeightedAverage = np.dot(arr1,arr2) # numerator of weighted average
    denominatorWeightedAverage = sum(finalUserVector)
    predictedRating = numeratorWeightedAverage/denominatorWeightedAverage # compute predicted rating
    #print(rating)
    #print(sum(finalUserVector))
    print("Movie rating by user {} for movie {} is predicted to be {}".format(userId,movieId,predictedRating))
    
    
# Local path to the data files
movies_url = "C:\\Users\\Prthamesh\\Downloads\\ml-25m\\OldData2\\movies.csv"
ratings_url = "C:\\Users\\Prthamesh\\Downloads\\ml-25m\\OldData2\\ratings.csv"

# read csv files into pandas dataframe
movies_df = pd.read_csv(movies_url,usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})
rating_df=pd.read_csv(ratings_url,usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# Merge both dataframes on movieId. movieId is common attribute to both dataframes
merged_df = pd.merge(rating_df,movies_df,on='movieId')

movie_ratings_combined = merged_df.dropna(axis = 0, subset = ['title']) # We drop nan values here.
movie_rating_Counts = (movie_ratings_combined.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    ) # This tells us how many ratings are there for every movie

#We merge original dataframe with rating counts
rating_including_total_Rating_Counts = movie_ratings_combined.merge(movie_rating_Counts, left_on = 'title', right_on = 'title', how = 'left')

# We can discard the less popular movies by setting popularity_threshold
popularity_threshold = 0
rating_popular_movie= rating_including_total_Rating_Counts.query('totalRatingCount >= @popularity_threshold')

#Now we create a Pivot matrix
movie_features_df=rating_popular_movie.pivot_table(index='userId',columns='title',values='rating').fillna(0)

# First parameter - userId
# Second parameter - movieId
# Third parameter - value of k.
predictMovieRating(35,224,6)


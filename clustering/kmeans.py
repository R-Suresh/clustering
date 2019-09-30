from collections import defaultdict
from math import inf
import math
import numpy as np
import random
import csv
import pandas as pd


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    mean=np.zeros_like(points[0])
    for point in points:
        mean+=point
    mean=mean/len(points)
    return mean
    # raise NotImplementedError()


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """

    centers=[]
    for label in np.unique(assignments):
        cluster=[]
        for i, assignment in enumerate(assignments):
            if(assignment==label):
                cluster.append(data_set[i])
        centers.append(point_avg(cluster))
    # return the center for each group
    return centers
    # raise NotImplementedError()


def assign_points(data_points, centers):
    """
    Input : data points and the array representing centres for all the clusters
    Output : returns the assigment for each point in the dataset
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    return math.sqrt(np.sum(np.square(np.array(a)-np.array(b))))
    # raise NotImplementedError()


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    chosen_points = [data_set[random.randint(0, len(data_set) - 1)] for y in range(1, k)]
    return chosen_points
    # raise NotImplementedError()


def get_list_from_dataset_file(dataset_file):
    '''
    param dataset_file: file whci represents the dataset
    return: returns data points from the file
    '''
    df=pd.read_csv(dataset_file)
    return df.values
    # raise NotImplementedError()


def cost_function(clustering):
    '''
    param clustering: dict of assignments and points
    return: the mean euclidean loss for the entire clustering
    '''
    points=[]
    assignments=[]
    for key, values in clustering.items():
        for value in values:
            assignments.append(key)
            points.append(value)
        centers=update_centers(points,assignments)
    cost=0
    for i,point in enumerate(points):
        cost+=distance(point,centers[assignments[i]-1])
    return cost/len(points)
    # raise NotImplementedError()


def k_means(dataset_file, k):
    dataset = get_list_from_dataset_file(dataset_file)
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering

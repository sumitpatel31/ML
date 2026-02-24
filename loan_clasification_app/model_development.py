import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

def dt( ):
    dt=DecisionTreeClassifier()
    return dt
def nb():
    nb=GaussianNB()
    return nb
def knn():
    knn=KNeighborsClassifier()
    return knn
def lr():
    lr=LogisticRegression()
    return lr
def rf():
    rf=RandomForestClassifier()
    return rf
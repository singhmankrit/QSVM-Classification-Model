# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:30:24 2020

@author: Mankrit
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import qiskit
from qiskit import Aer
from qiskit import IBMQ
from qiskit import BasicAer
from qiskit.ml.datasets import *
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit import QuantumCircuit, execute
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel

#preprocessing
train = pd.read_feather('data5_lumAB_train_normalized.feather')
test = pd.read_feather('data5_lumAB_test_normalized.feather')

#doing PCA
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from sklearn.model_selection import cross_val_predict

pca = PCA(n_components = 10)
transform1 = Pipeline([("pca", pca),])
transform2 = Pipeline([("pca", pca),])

X_train = train.iloc[:, 1:].values
y_train = train.loc[:,'cancer'].values

X_test = test.iloc[:, 1:].values
y_test = test.loc[:,'cancer'].values

X_train = transform1.fit_transform(X_train)
X_test = transform2.transform(X_test)

y_train = y_train.reshape(250,1)
y_test = y_test.reshape(61,1)
train = np.append(X_train, y_train, axis=1)
test = np.append(X_test, y_test, axis=1)

train = pd.DataFrame(data=train[:,:])
test = pd.DataFrame(data=test[:,:])
train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)
train = {k: np.asarray(g.iloc[:, :-1]) for k,g in train.groupby(10)}
test = {k: np.asarray(g.iloc[:, :-1]) for k,g in test.groupby(10)}

backend = BasicAer.get_backend('qasm_simulator')

#IBMQ.enable_account(HERE)
#provider = IBMQ.get_provider(hub='ibm-q')
#backend = provider.get_backend('ibmq_16_melbourne') # Specifying Quantum device

seed = 11111
feature_dim = 10
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')
qsvm = QSVM(feature_map, train, test)

quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed, skip_qobj_validation=False)

result = qsvm.run(quantum_instance)
print("Quantum accuracy on test set: {0}%".format(round(result['testing_accuracy']*100, 2)))
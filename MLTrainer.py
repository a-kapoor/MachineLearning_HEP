#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import os
import sys
print("Packages Loaded from "+sys.executable)
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optparse, json, argparse, math
import ROOT
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Reshape,Conv2D,MaxPooling2D,ConvLSTM2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from root_numpy import root2array, tree2array

#import Config as Conf
#import ConfigMultiClass as Conf
seed = 7
##small changes
np.random.seed(7)
rng = np.random.RandomState(31337)
timestr=time.strftime("%Y%m%d-%H%M%S")
#from tqdm import tqdm

def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
    
def load_data(inputPath,variables,criteriaf,keysf,sampleNamesf,fileNamesf,targetf):
    # Load dataset to .csv format file
    my_cols_list=variables+['process', 'key', 'target', 'sampleWeight']
    data = pd.DataFrame(columns=my_cols_list)

    for key in keysf :
        print(key)
        sampleNames=sampleNamesf[key]
        fileNames = fileNamesf[key]
        target=targetf[key]
        #criteria=criteriaf[key]
        inputTree = 'Friends'
        print(sampleNames)

        for process_index in range(len(fileNames)):
            fileName = fileNames[process_index]
            sampleName = sampleNames[process_index]

            try: tfile = ROOT.TFile(inputPath+"/"+fileName+".root")
            except :
                print(" file "+ inputPath+"/"+fileName+".root doesn't exits ")
                continue
            try: tree = tfile.Get(inputTree)
            except :
                print(inputTree + " deosn't exists in " + inputPath+"/"+fileName+".root")
                continue
            if tree is not None :
                print('criteria: ', criteriaf[process_index])
                #try: chunk_arr = tree2array(tree=tree, selection=criteria, start=0, stop=100) # Can use  start=first entry, stop = final entry desired
                try: chunk_arr = tree2array(tree=tree, selection=criteriaf[process_index]) # Can use  start=first entry, stop = final entry desired
                except : continue
                else :
                    chunk_df = pd.DataFrame(chunk_arr, columns=variables)
                    chunk_df['process']=sampleName
                    chunk_df['key']=key
                    chunk_df['target']=target
            data = data.append(chunk_df, ignore_index=True)
            tfile.Close()
        if len(data) == 0 : continue
        processfreq = data.groupby('key')
        samplefreq = data.groupby('process')         
        print("TotalWeights = %f" % (data.iloc[(data.key.values==key)]["sampleWeight"].sum()))
        nNW = len(data.iloc[(data["sampleWeight"].values < 0) & (data.key.values==key) ])
        print(key, "events with -ve weights", nNW)
    print('<load_data> data columns: ', (data.columns.values.tolist()))
    n = len(data)
    return data

def MakePlots(y_train, y_test, y_test_pred, y_train_pred, Wt_train, Wt_test,ROCMask,keys,od,keycolor):
    from sklearn.metrics import roc_curve, auc
    fig, axes = plt.subplots(1, len(keys), figsize=(5*len(keys),5))

    figMVA, axesMVA = plt.subplots(1, len(keys), figsize=(5*len(keys), 5))

    for i in range(len(keys)):
        print(i)
        ax=axes[i]
        axMVA=axesMVA[i]
        nodename=keys
        trainkeys = [x + '_train' for x in keys]
        testkeys = [x + '_test' for x in keys]
        for j in range(len(keys)):
            axMVA.hist(y_test_pred[:, i][(y_test[:, j]==1)],
                       bins=np.linspace(0, 1, 21),label=testkeys[j],
                       weights=Wt_test[(y_test[:, j]==1)]/np.sum(Wt_test[(y_test[:, j]==1)]),
                       histtype='step',linewidth=4,color=keycolor[j])
            axMVA.hist(y_train_pred[:, i][(y_train[:, j]==1)],
                       bins=np.linspace(0, 1, 21),label=trainkeys[j],
                       weights=Wt_train[(y_train[:, j]==1)]/np.sum(Wt_train[(y_train[:, j]==1)]),
                       histtype='stepfilled',alpha=0.2,linewidth=1,color=keycolor[j])
            axMVA.set_title('MVA: Node '+str(nodename[i]),fontsize=20)
            axMVA.legend(loc="upper right",fontsize=10)
            axMVA.set_xlim([0, 1])
        
        fpr, tpr, th = roc_curve(y_test[:, i], y_test_pred[:, i],sample_weight=Wt_test)
        fpr_tr, tpr_tr, th_tr = roc_curve(y_train[:, i], y_train_pred[:, i],sample_weight=Wt_train)
        mask1 = tpr > ROCMask
        fpr, tpr = fpr[mask1], tpr[mask1]
    
        mask2 = tpr_tr > ROCMask
        fpr_tr, tpr_tr = fpr_tr[mask2], tpr_tr[mask2]
    
        roc_auc = auc(fpr, tpr)
        roc_auc_tr = auc(fpr_tr, tpr_tr)
    
        ax.plot(tpr, 1-fpr, label='ROC curve test (area = %0.2f)' % roc_auc,linewidth=4)
        ax.plot(tpr_tr, 1-fpr_tr, label='ROC curve train (area = %0.2f)' % roc_auc_tr,linewidth=4)
        #plt.plot([0, 1], [0, 1], 'k--')
        #ax.set_xlim([0.8, 1.0])
        #ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Signal efficiency',fontsize=20)
        ax.set_ylabel('Background rejection',fontsize=20)
        ax.set_title('ROC: Node '+str(nodename[i]),fontsize=20)
        #ax.set_yscale("log", nonposy='clip')
        ax.legend(loc="upper left",fontsize=10)
    fig.savefig(od+"/ROC"+timestr+".pdf")
    figMVA.savefig(od+"/output"+timestr+".pdf")

def in_ipynb():
    try:
        cfg = get_ipython().config
        print(cfg)
        if 'jupyter' in cfg['IPKernelApp']['connection_file']:
            return True
        else:
            return False
    except NameError:
        return False


if in_ipynb(): 
    print("In IPython")
    exec("import ConfigMultiClass as Conf")
    TrainConfig="ConfigMultiClass"
else:
    TrainConfig=sys.argv[1]
    prGreen("Importing settings from "+ TrainConfig.replace("/", "."))
    #exec("from "+TrainConfig+" import *")
    importConfig=TrainConfig.replace("/", ".")
    exec("import "+importConfig+" as Conf")


# In[2]:


def load_trained_model(weights_path, num_variables, optimizer,nClasses):
    model = baseline_model(num_variables, optimizer,nClasses)
    model.load_weights(weights_path)
    return model

def normalise(x_train, x_test):
    mu = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_normalised = (x_train - mu) / std
    x_test_normalised = (x_test - mu) / std
    return x_train_normalised, x_test_normalised

def check_dir(dir):
    if not os.path.exists(dir):
        print('mkdir: ', dir)
        os.makedirs(dir)

# Ratio always > 1. mu use in natural log multiplied into ratio. Keep mu above 1 to avoid class weights going negative.
def create_class_weight(labels_dict,mu=0.9):
    total = np.sum(list(labels_dict.values())) # total number of examples in all datasets
    keys = list(labels_dict.keys()) # labels
    class_weight = dict()
    print('total: ', total)

    for key in keys:
        # logarithm smooths the weights for very imbalanced classes.
        score = math.log(mu*total/float(labels_dict[key])) # natlog(parameter * total number of examples / number of examples for given label)
        #score = float(total/labels_dict[key])
        print('score = ', score)
        if score > 0.:
            class_weight[key] = score
        else :
            class_weight[key] = 1.
    return class_weight


# In[2]:


# In[3]:


print('Using Keras version: ', tf.keras.__version__)

do_model_fit = 1
classweights_name = 'InverseSRYields'#args.classweights''
selection = 'th'#args.selection''
# Number of classes to use
number_of_classes = Conf.NClass
# Create instance of output directory where all results are saved.
output_directory = Conf.output_directory
od = output_directory

check_dir(output_directory)

# Create plots subdirectory
plots_dir = os.path.join(output_directory,'plots/'+timestr)
plots_dir3 = os.path.join(output_directory,'plots3/')
plots_dir4 = os.path.join(output_directory,'plots4/')

#input_var_jsonFile = open('input_vars_SigRegion_wFwdJet.json','r')
# input_var_jsonFile = open('inputfor2lss.json','r')
# input_var_jsonFile1 = open('inputfor2lss1.json','r')

selection_criteria1 = Conf.Train_selection_criteria
selection_criteria2 = Conf.Test_selection_criteria

criteria1=[selection_criteria1+" & "+Conf.selections[key] for key in Conf.keys]
criteria2=[selection_criteria2+" & "+Conf.selections[key] for key in Conf.keys]

#Before split
variable_list1 = Conf.Varlist
#After split
variable_list = Conf.Varlist

# Create list of headers for dataset .csv
column_headers = []
column_headers1 = []
for key in variable_list:
    column_headers.append(key)
for key in variable_list1:
    column_headers1.append(key)
    
print(column_headers)
print(column_headers1)

# Create instance of the input files directory
inputs_file_path = Conf.inputs_file_path

#'#/publicfs/cms/data/TopQuark/cms13TeV/ForDNNSharing/sample'

# Load ttree into .csv including all variables listed in column_headers
print('<train-DNN> Input file path: ', inputs_file_path)
#outputdataframe_name = '%s/output_dataframe_%s.csv' %(output_directory,selection)
outputdataframe_nametr = '%s/output_dataframe_tr_%s.csv' %(output_directory,selection)
outputdataframe_namete = '%s/output_dataframe_te_%s.csv' %(output_directory,selection)

print('<train-DNN> Creating new data .csv @: %s . . . . ' % (inputs_file_path))
datatr = load_data(inputs_file_path,column_headers1,criteria1,keysf=Conf.keys,sampleNamesf=Conf.sampleNames,fileNamesf=Conf.fileNames,targetf=Conf.target)
datate = load_data(inputs_file_path,column_headers1,criteria2,keysf=Conf.keys,sampleNamesf=Conf.sampleNames,fileNamesf=Conf.fileNames,targetf=Conf.target)

print(datatr.head())


# In[4]:


#Extra constant

datatr['sampleWeightDNN']=1
datate['sampleWeightDNN']=1
datatr['sampleWeightROC']=1
datate['sampleWeightROC']=1

for key,target in zip(Conf.keys,Conf.target):
    datatr.loc[datatr.target == target, 'sampleWeightDNN']= Conf.sampleWeightDNN[key]
    #datatr.loc[datatr.target == target, 'sampleWeightDNN']=datatr.loc[datatr.target == target, 'sampleWeightDNN'] / datatr.loc[datatr.target == target, 'sampleWeightDNN'].sum()
    datate.loc[datate.target == target, 'sampleWeightDNN']= Conf.sampleWeightDNN[key]
    datatr.loc[datatr.target == target, 'sampleWeightROC']= Conf.sampleWeightROC[key]
    #datatr.loc[datatr.target == target, 'sampleWeightROC']=datatr.loc[datatr.target == target, 'sampleWeightROC'] / datatr.loc[datatr.target == target, 'sampleWeightROC'].sum()
    datate.loc[datate.target == target, 'sampleWeightROC']= Conf.sampleWeightROC[key]

# Create statistically independant lists train/test data (used to train/evaluate the network)
#traindataset, valdataset = train_test_split(data, test_size=0.2)
traindataset =datatr.copy()
valdataset=datate.copy()
#valdataset.to_csv('valid_dataset.csv', index=False)
training_columns = column_headers
print('<train-DNN> Training features: ', training_columns)

# Select data from columns under the remaining column headers in traindataset
X_train = traindataset[training_columns].astype('float32')
print("Training events:"+str(len(traindataset)))
print("Training events:"+str(len(traindataset[training_columns])))
print("Training events:"+str(len(X_train)))
Y_train = traindataset.target.astype(int)
X_test = valdataset[training_columns].astype('float32')
Y_test = valdataset.target.astype(int)

num_variables = len(training_columns)

####################
trainweights = traindataset.loc[:,'sampleWeightDNN'] #Norm weight x xsec wt x (cpscaleweight)
trainweights = np.array(trainweights)

testweights = valdataset.loc[:,'sampleWeightDNN']
testweights = np.array(testweights)

train_weights = traindataset['sampleWeightDNN'].values
test_weights = valdataset['sampleWeightDNN'].values
###################

###################
trainweightsROC = traindataset.loc[:,'sampleWeightROC'] #Norm weight
trainweightsROC = np.array(trainweights)

testweightsROC = valdataset.loc[:,'sampleWeightROC']
testweightsROC = np.array(testweights)

train_weightsROC = traindataset['sampleWeightROC'].values
test_weightsROC = valdataset['sampleWeightROC'].values
###################

# Fit label encoder to Y_train
newencoder = LabelEncoder()
newencoder.fit(Y_train)
# Transform to encoded array
encoded_Y = newencoder.transform(Y_train)
encoded_Y_test = newencoder.transform(Y_test)
# Transform to one hot encoded arrays
# Y_train = np_utils.to_categorical(encoded_Y)
# Y_test = np_utils.to_categorical(encoded_Y_test)
Y_train = to_categorical(encoded_Y)
Y_test = to_categorical(encoded_Y_test)
optimizer = 'Adam'#'Nadam'
if do_model_fit == 1:
    histories = []
    labels = []
    # Define model and early stopping
    model = Conf.mymodel(num_variables,optimizer,number_of_classes)
    DNNDict=Conf.DNNDict
    if DNNDict['earlyStop']:
        es = EarlyStopping(patience=10,monitor='val_loss',verbose=1)
        history = model.fit(X_train,Y_train,validation_data=(X_test, Y_test, testweights), epochs=DNNDict['epochs'], batch_size=DNNDict['batch_size'], verbose=1, shuffle=True, sample_weight=trainweights, callbacks=[es])
    else:
        history = model.fit(X_train,Y_train,validation_data=(X_test, Y_test, testweights), epochs=DNNDict['epochs'], batch_size=DNNDict['batch_size'], verbose=1, shuffle=True, sample_weight=trainweights)
    #model3 = newCNN_model(num_variables,optimizer,number_of_classes,1000,0.40)
    
    histories.append(history)
    labels.append(optimizer)                                                                                                                                                                            
else:
    # Which model do you want to load?                  
    model_name = 'model.h5'
    print('<train-DNN> Loaded Model: %s' % (model_name))
    model = load_trained_model(model_name,num_variables,optimizer,number_of_classes)
# Node probabilities for training sample events
result_probs = model.predict(np.array(X_train))
result_classes = model.predict_classes(np.array(X_train))

# Node probabilities for testing sample events     
result_probs_test = model.predict(np.array(X_test))
result_classes_test = model.predict_classes(np.array(X_test))
# Store model in file                                        
model_output_name =od+'/model.h5'
model.save(model_output_name)
weights_output_name = od+'/model_weights.h5'
model.save_weights(weights_output_name)
model_json = model.to_json()
model_json_name = od+'/model_serialised.json'
with open(model_json_name,'w') as json_file:
    json_file.write(model_json)
model.summary()
#model_schematic_name = os.path.join(output_directory,od+'/model_schematic.pdf')
#plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True)
    


# In[5]:


# In[18]:
MakePlots(y_train=Y_train, y_test=Y_test, y_test_pred=result_probs_test, y_train_pred=result_probs, Wt_train=train_weightsROC, Wt_test=test_weightsROC,ROCMask=Conf.ROCMask,keys=Conf.keys,od=Conf.output_directory,keycolor=Conf.keycolor)


# In[ ]:





# In[6]:


#datatr["predTarget"] = [list(p).index(max(p)) for p in model.predict(np.array(X_train))]
#datate["predTarget"] = [list(p).index(max(p)) for p in model.predict(np.array(X_test))]


# In[7]:


datatr["predTarget"] = result_classes
datate["predTarget"] = result_classes_test


# In[8]:


import seaborn as sns
confusion_matrix = pd.crosstab(datatr["target"], datatr["predTarget"], rownames=['Actual'], colnames=['Predicted'])
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
sns_plot=sns.heatmap(confusion_matrix,cmap="YlGnBu", annot=True, cbar=False,fmt='g',ax=axes)
plt.savefig(od+"/confusion_matrix_train.png")

confusion_matrix = pd.crosstab(datate["target"], datate["predTarget"], rownames=['Actual'], colnames=['Predicted'])
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
sns_plot=sns.heatmap(confusion_matrix,cmap="YlGnBu", annot=True, cbar=False,fmt='g',ax=axes)
plt.savefig(od+"/confusion_matrix_test.png")


# In[9]:


original_stdout = sys.stdout
from sklearn import metrics
with open(od+'/ClassificationReport.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("-----------------")
    print("Train Dataset Classification Report")
    print(metrics.classification_report(traindataset.target.astype(int), result_classes,target_names=Conf.keys))
    print("-----------------")
    print("Test Dataset Classification Report")
    print(metrics.classification_report(valdataset.target.astype(int), result_classes_test,target_names=Conf.keys))
    print("-----------------")
    sys.stdout = original_stdout


# In[ ]:





# In[ ]:





# In[ ]:





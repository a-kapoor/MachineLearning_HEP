
Varlist=["Lep1_pt","Lep2_pt","Lep1_eta","Lep2_eta","Lep1_phi","Lep2_phi","mT_lep2","mT_lep1","mindr_lep1_jet","mindr_lep2_jet","mTTH_2lss","dEtaBB_2lss","dEtaLL_BBframe_2lss","dEtaBB_LLframe_2lss","avg_dr_jet","nSelJets","met","met_phi"]


Train_selection_criteria = 'Entry$%2==0' #Even events
Test_selection_criteria = 'Entry$%2!=0' #Odd events

NClass=3 #Multi Classification

output_directory='./result/'

inputs_file_path = '/eos/user/a/akapoor/SWAN_projects/ttHCPnewStrategy/' #Where are the input files?

keys=['ttH','ttJ','ttW']
keycolor=['red','blue','green']

sampleNames={'ttH':['ttH'],
             'ttJ':['ttJ'],
             'ttW':['ttW']} #Names of process (Can be same as keys)


fileNames={'ttH':['TTH_ctcvcp_new_Friend_Run2'],
           'ttJ':['TTJets_DiLepton_Friend_Run2'],
           'ttW':['TTWToLNu_fxfx_Friend_Run2']} #File names without the ".root"

target={'ttH':0,'ttJ':1,'ttW':2}

sampleWeightDNN={'ttH':1,'ttJ':1,'ttW':1} #Will go to DNN loss
sampleWeightROC={'ttH':1,'ttJ':1,'ttW':1} #Will be used while plotting ROC

ROCMask = 0.7 #ROC plot will start at this signal eff

DNNDict={'epochs':5, 'batch_size':1000, 'earlyStop':True}

def mymodel(num_variables,optimizer,nClasses):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,Reshape,Activation,Flatten,Dropout,BatchNormalization
    from tensorflow.keras.optimizers import Adam,Nadam
    model = Sequential()
    model.add(Dense(num_variables*2,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Dense(num_variables,activation='relu'))
    model.add(Dense(num_variables,activation='relu'))
    model.add(Dense(num_variables,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nClasses, activation='softmax'))
    if optimizer=='Adam':
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['acc'])
    if optimizer=='Nadam':
        model.compile(loss='categorical_crossentropy',optimizer=Nadam(lr=0.001),metrics=['acc'])
    return model
    

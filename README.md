# MachineLearning_HEP

### Clone

```
git clone https://github.com/akapoorcern/MachineLearning_HEP.git

```

### Setup

```
source /cvmfs/sft.cern.ch/lcg/views/LCG_97python3/x86_64-centos7-gcc8-opt/setup.sh

```

### All you need to do is to edit the Config file with the settings for your analysis


#### Example for Binary Classfication is ConfigBinary.py 
#### and for multiclass is ConfigMultiClass.py

### and then run 

``` 
python MLTrainer-MultiClass.py <ConfigName> #without the .py on config

```
####  example

```
python MLTrainer-MultiClass.py ConfigMultiClass

```

### The MLTrainer will read the settings from the config file and run training
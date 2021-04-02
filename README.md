# MachineLearning Package to do both binary and multi-class classification

### Clone

```
git clone https://github.com/akapoorcern/MachineLearning_HEP.git

```

### Setup

```
source /cvmfs/sft.cern.ch/lcg/views/LCG_97python3/x86_64-centos7-gcc8-opt/setup.sh

```
### Create a new custom Config file (Just copy the original and start editing on top of it)

```

cp Config.py MyNewConfig.py

```

### All you need to do is to edit the Config file with the settings for your analysis

### and then run 

``` 
python MLTrainer.py <ConfigName> #without the .py on config

```
####  example

```
python MLTrainer.py MyNewConfig

```

### The MLTrainer will read the settings from the config file and run training
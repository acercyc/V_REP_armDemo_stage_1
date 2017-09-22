# V-Rep Data Generation

armDemo_vision_data_gen.py

​	=> /armDemo_vision_data_gen

​		=>**Full set** (10000 pics)

​			run_single_2

​			run_multi_1

# Training object center detection model (stage 1)

* armDemo\_stage1\_loadData.py
* armDemo_stage\_1\_model\_2.py
  \# **Single** object model
* armDemo_stage\_1\_model\_3_multi.py
  \# Multi-object model
* armDemo_stage\_1\_model\_4_multi.py
  * Multi-object model with high prediction
  * gaussFilter = 4
* armDemo_stage\_1\_model\_4\_2\_multi.py
  * Same as armDemo_stage\_1\_model\_4_multi.py just with gaussFilter = 2
  * This is the current final model


##  Output folder

/armDemo_stage1

​	model\_2\_CModelCheckpoint\_best.hdf5

​	model\_2\_CSVLogger.csv

​	model\_3\_CModelCheckpoint\_best.hdf5

​	model\_3\_CSVLogger.csv



# First demo

armDemo\_stage1\_integration2.py

​	\# The first complete local version of this demo





# Others

V_REP_armDemo_stage1\_localVersion
# README for network.py, data.py and kinematic.py

This is a Hebbian Learning model and software for train/test data creation. It is supposed to learn representation of joint angles and posititions in 2D/3D.

network.py, data.py and kinematic.py are the three main files for simulation. In the following, their tasks and contents are briefly explained.

## network.py

=> creates and simulates the Hebbian network using ANNARCHY
=> gets train/test neural inputs from data.py

## data.py

=> where the train/test data is created using the methods defined in kinematic.py
=> the train/test data is then translated into neural input data

## kinematic.py

=> where the basic kinematic functions for deriving shoulder joint angle, elbow joint angle, horizontal angle of stimulus position to head center, and tactile information on forearm are calculated
=> based on denavit-hartenberg convention (see "./dh_kinematic/literature")

## Parameters

### link lengths in meters:

- shoulder_distance = 0.05
- upper_arm_length = 0.22
- forearm_length = 0.16

find angle limits in ./dh_kinematic/constraints.py

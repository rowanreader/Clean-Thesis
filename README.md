# Cobot

Motion Planning (Low Level) Agent

Agent is built in TestAI.py, and requires Environment.py, TowerSim.py, and SawyerSim.py to be in the same directory (main directory). 
TestAI.py is currently set to train the teleport agent from scratch. 
To run the IK simulation, follow directions at line 19 and line 42 in TestAI.py, and check that the correct step function is being used (the one not in use should be changed to step2) in Environment.py


The following is found in the HighLevel directory:

Tile Placement Agent (High Level):
The environment used for the tile placement agent can be found in placeTileEnv.py, and requires PlaceTileSim.py and HighTowerSim.py. The environment is automatically run by placeTileAI, which has the agent.
To train, run placeTileAI.py


Pillar Placement Agent (High Level):
Towers were generated and saved with MinHighSim.py, which required SawyerSim.py and PhysicsConnect.py. SawyerSim.py only provided the getGoal function, while PhysicsConnect.py was used to simulate the tower in Coppelia and placed pillars from the center of the tile out.
PhysicsConnect requires access to the Tile class from MinHighSim, to avoid circular imports, PhysicsConnect uses the Tile class provided by HighTowerSim.py
Note: HighTowerSim.py was the original attempt to build towers using an algorithm. It is not used for anything except the Tile class currently, and a few other base functions

The environment used is in HighEnvs.py, and is automatically run by HighAI1.py and DQN2. It requires CoppeliaTower.py, placeTileEnv.py, and HighTowerSim.py
HighAI1.py uses the stable baselines AI (traditional DQN).
DQN2 builds a DQN from scratch and is set up to make both a traditional DQN and an Attention based DQN.
Note: there is another file, DQNHigh1.py, which was set up to use the autoencoder. This method did not work and was discarded. It has been included for the sake of completion.
To turn off simulation, please follow instructions at line 298, line 311 (which directs to line 258), line 318 in HighEnvs.py
CoppleilaTower.py is different from PhysicsConnect.py in that it takes in the initial tower, which it builds in the same way as PhysicsConnect, and then a list of additional pieces, which it adds on in order of sequence. This is necessary to preserve the sequence of the agent's moves.

HighAlgorithms.py contains 2 algorithms - the pillar selection algorithm and the level difficulty algorithm. This file required HighTowerSim.py.

Pillar Selection Agent (Algorithm):
This algorithm was designed to select the best 5 pillars for the agent to have for a turn out of a possible 11.

Level Difficulty Agent (Algorithm):
This algorithm was responsible for determining how many pillars to place on a given turn - 1, 2, or 3.



Structure of the file:
	classifiers/
		GlobalFeatureVector/
			Protoss_vs_Protoss/
				Protoss/
					<replay_files>
			Protoss_vs_Terran
			Protoss_vs_Zerg
			Terran_vs_Terran
			Terran_vs_Zerg
			Zerg_vs_Zerg
		to_train/
			run_classifiers_auto.py
			<matchup>.csv
			<matchup>_frame<#frame>.csv
		parsing.py
		readme

/!\ This program does not take duplicates into account. Before creating files with the scripts, make sure the filename is available.

There are two scripts to use to get the classifiers.
The entire dataset is not available here, only 'Protoss_vs_Protoss' can be used.
To extract the vectors that hold the information for a given frame (minerals, vespene, army count, etc..), use 'parsing.py'
Once this pre-processing is done, use 'run_classifiers_auto.py' to test and get the best classifiers for a given matchup.

- parsing.py
	line 14	: Select the matchup you want to get the info out of 
		(only 'PvP' is available here, if you would like, contact us to get the rest of the data)
	line 50	: Modify the list to change the frames being analyzed. 
	
	The output of 'parsing.py' will be the '<matchup>_frame<#frame>.csv' files, in the main/to_train/ folder.

- run_classifiers_auto.py
	line 18 : Change the value to the desired matchup (All matchups are available here)

	'run_classifiers_auto.py' will use the '<matchup>_frame<#frame>.csv' files as input and ouput <matchup>.csv

- <matchup>.csv
	1st line : Type of classifier
	n-th line : n-th value of frame in frame's list
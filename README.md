# sc2_win_predict
Random Forest classifiers to determine win rate at different points in any SC2 game and associated front-end that allows replay analysis and win rate during replays.

The current directory needs to be the folder where this readme exists,
where manage.py and requirements.txt.
Python 3 was used to develop the app and will be needed to run this version of django 2.0

To install the required libraries, the following command can be run:
pip install requirements.txt
If both python 2 and 3 are installed, pip3 install requirements.txt

To run the app just run the following command from cmd/terminal:
python manage.py runserver

If both python 2 and 3 are installed,
Windows: py -3 manage.py runserver
Linux/osx: python3 manage.py runserver

This will create a local host. The website can be viewed by going to any of the following:
1. http://127.0.0.1:8000
2. http://localhost:8000
3. The server mentioned once launched inside the cmd/terminal

The slider represents current frame number allowing the user to select a different frame.
Pressing "Load Time" loads the frame.
Pressing "Reset" defaults to the first frame of the first file and displays its output.
Pressing "Load File" selects the file number displayed and displays its output.
Pressing "Upload" uploads the selected Global Feature Vector and Spatial Feature Tensor and displays its output.
An invalid input (i.e. 1>filenumber>16) and pressing the load button will be thresholded to this constraint and not generate an error.


The app provides two options to select the file. The user can select one of the 16 replays in the 
subsidized dataset or upload a replay (Spatial Feature Tensor and Global Feature Vector) as desired.
The app will then use the corresponding input to generate a prediction, actual result of the match 
using just the current frame. A slider is included which allows the user to go back and forth in terms
of the current frame and use a different frame in the replay to generate the image and predictions.


The app uses a django frontend with scikit-learn in the backend. The time instance and the 
input file or file selected is used to generate the image using matplotlib and prediction
using the scikit-learn's pre-trained random forest model. A tiny subset of the dataset, specifically
16 replays, are provided with the Spatial Feature Tensor and corresponding Global Feature
Vector.



*********************************************************************************
FOR 'classifiers and evaluation/classifiers':
The folder classsifiers contain classifier implementations including training and evaluation.
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


*******************************************************************************
FOR 'classifiers and evaluation'
This folder contains a random forest that works for all time instances. The files need to be copied into their parent directory to have the correct path.
check_accuracy will evaluate the model training it on 1000 files of the training dataset and testing it on the entire dataset depending on the competition specified.
All the files here use the test_train split json files specified in the msc dataset so the entire dataset (Global Feature Vectors) needs to be downloaded and added to path before this can be run.

# ImageClassification

---------------------------------------------------------------------------------------------------------------
Creating the Training Dataset
1. The code folder must contain a ‘train’ folder, which contains sub folders named ‘cat’ and ‘dog’.
2. Each of the subfolder should contain 12,500 images of cats and dogs from the dataset.

--------------------------------------------------------------------------------------------------------------
Creating the Test Dataset
1. Create another folder named ‘test1’ containing images from the test1.zip folder

--------------------------------------------------------------------------------------------------------------
Steps to Train the Data:
1. Run generateTrainingData, using the command:
```
python3 generateTrainingData.py
```

2. Now one the training data is created into the respective pickel format, Run the TrainModel file using the command
```
python3 trainModel.py
```

-------------------------------------------------------------------------------------------------------------
Testing the Generated model on the Test Data:
1. The latest model will be generated in the same folder; Now copy-paste the name of the latest model (ex. Classification-blahblah.h5) into the classifyImages.py file for variable ‘name’.

2. Now run the file classifyImages.py using the command to generate the results using the trained model.
```
python3 classifyImages.py
```

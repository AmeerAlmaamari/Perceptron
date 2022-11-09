To run the attached codes you need the following packages:

numpy==1.20.1
pandas==1.2.4
matplotlib==3.3.4
seaborn==0.11.1
tqdm==4.59.0
---------------------------------

To open the datasets you need to change the path in the code, and import them with pandas

  train = pd.read_csv(r".../FILE PATH")
  test= pd.read_csv(r".../"

--------------------------------------------------------------
Also, to train or test certain classes you need to:

1- in train and test functions calls, change the parameter to the desired classes
e.g. train_12, X_test_12 etc...

2- To test accuracies, just change the parameter of the called function (accuracy)
e.g. y_pred_12, y_true_12 etc...

** other labels for true and predicted will be commented as #y_pred_23 and #y_pred_13
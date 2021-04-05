Module Sklearn
==============
<a href="https://docs.zetane.com/getting_started.html"><button type="button" name="button" style="color:#fff; background-color:#5E9A5C; border-radius:10px; width: 200px; height: 50px;margin-left:85%;font-size:20px;margin-right:auto">Download<br /> Zetane Viewer</button></a>
<br />
<br />
<br />

<img align="left" width="500" height="250" src = "../images/sklearn2onnx_ztn_view.png" style="margin-right: 2000px;">

**DESCRIPTION**: This script used to classification of the iris dataset using RandomForestClassifier from sklearn 

First the training dataset and test dataset is separated using train_test_split() function of sklearn.model_selection 

Then the training data is fitted to the random forest model 

    clr = RandomForestClassifier() 

    clr.fit(X_train, y_train) 

The trained model is stored in onnx format using skl2onnx library 

    onx = convert_sklearn(clr, initial_types=initial_type, options={type(clr): {'zipmap': False}}) 

    with open("rf_iris.onnx", "wb") as f: 

        f.write(onx.SerializeToString()) 

Model is loaded in the zetane engine using the model() function inside the Context module 

    zcontext = ztn.Context() 

    zonnx = zcontext.model().onnx(model).update(X_test.astype(numpy.float32)) 

Model is saved using the save() function 

    zcontext.save(filename+'/Sklearn2Onnx.ztn') 

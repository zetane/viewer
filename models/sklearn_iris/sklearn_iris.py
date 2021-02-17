import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import zetane as ztn 
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

clr = RandomForestClassifier()
clr.fit(X_train, y_train)

# Convert into ONNX format
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
directory = os.path.dirname(__file__)
#filename = os.path.join(directory, 'Sklearn2Onnx')
initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type, options={type(clr): {'zipmap': False}})
with open(directory+r"\rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Compute the prediction with ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession(directory+r"\rf_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
pred_onx


zcontext = ztn.Context()
zcontext.clear_universe()
zmodel = zcontext.model()
model = directory+r"\rf_iris.onnx"
zonnx = zcontext.model().onnx(model).update(X_test.astype(numpy.float32))
zonnx.update()



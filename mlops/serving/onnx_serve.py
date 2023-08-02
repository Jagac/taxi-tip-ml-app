import onnx
import onnxruntime as rt
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from tagiftip import main

# requires sklearn < 1.3
# model is trained with > 1.3 which may cause mlflow dependecy issues
# this way uses the tuned hyperparams to quickly train and serve using onnx

df = main.load_data()


X = df.drop("tipped", axis=1).to_numpy()
y = df.tipped.to_numpy()

model = xgb.XGBClassifier(
    colsample_bytree=0.13417417512177854,
    gamma=0.3535128162952119,
    learning_rate=0.820235985660056,
    max_depth=6,
    min_child_weight=3,
    n_estimators=323,
    reg_alpha=0.5084254400098561,
    reg_lambda=0.5610049110562496,
    subsample=0.3267823046570212,
    n_jobs=-1,
)


model.fit(X, y)
onnx_model_path = (
    "/home/jagac/projects/taxi-tip-mlapp/mlops/serving/xgb_tip_no_tip.onnx"
)

initial_type = [("float_input", FloatTensorType([None, 17]))]

onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=10)
onnx.save(onnx_model, onnx_model_path)

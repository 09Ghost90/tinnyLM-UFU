import onnx 
from onnx_tf.backend import prepare

# Modelo ONNX
onnx_model = onnx.load("tiny_lm.onnx")

# ONNX -> TensorFlow
tf_rep = prepare(onnx_model)

# Salvar modelo TensorFlow
tf_rep.export_graph("tiny_lm_tf")
print("Modelo ONNX convertido e salvo como TensorFlow com sucesso.")
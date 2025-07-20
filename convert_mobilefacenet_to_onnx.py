import tensorflow as tf
import tf2onnx
import os

from src.config import MOBILEFACENET_MODEL_PATH

if __name__ == '__main__':
    # Load the TensorFlow model
    model_path = os.path.join(MOBILEFACENET_MODEL_PATH, "MobileFaceNet_9925_9680.pb")
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    # Convert the model to ONNX
    onnx_model, _ = tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=['input:0'],
        output_names=['embeddings:0'],
        opset=13
    )

    # Save the ONNX model
    onnx_path = os.path.join(MOBILEFACENET_MODEL_PATH, "mobilefacenet.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"MobileFaceNet model converted to ONNX and saved at {onnx_path}")

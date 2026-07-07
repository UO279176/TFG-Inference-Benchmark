import tensorflow as tf
import numpy as np

INPUT_SHAPE = (800, 800, 3)  # Height, Width, Channels
BATCH_SIZE = 1
STATIC_INPUT_SHAPE = (BATCH_SIZE,) + INPUT_SHAPE

# Load the .pb folder using Keras 3 TFSMLayer
model_dir = 'src/data/models/retinanet'
tf_sm_layer = tf.keras.layers.TFSMLayer(model_dir, call_endpoint='serving_default')

# Reconstruct a static Keras Functional Model
inputs = tf.keras.Input(shape=INPUT_SHAPE, batch_size=BATCH_SIZE)
outputs = tf_sm_layer(inputs)
model = tf.keras.Model(inputs, outputs)

print("Static input shape:", model.input.shape)

def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 800, 800, 3) * 255.0
        yield [data.astype(np.float32)]

"""
## Quantization
"""

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_dataset
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

"""
## Save the quantized model
"""

with open('quant_model.tflite', 'wb') as f:
    f.write(tflite_model)
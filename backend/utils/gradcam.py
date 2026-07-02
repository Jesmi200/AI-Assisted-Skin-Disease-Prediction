"""
gradcam.py
----------
Custom Keras layers (ECABlock, MSAM) used by the trained EfficientNetV2-S +
MSAM model, plus Grad-CAM utilities to visualize what the model "looked at"
before and after the MSAM attention block.

These layer definitions must match the ones used during training
(EfficientNetV2_MSAM.ipynb) exactly so that tf.keras.models.load_model()
can reconstruct the saved .keras model via custom_objects.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# --------------------------------------------------------------------------
# Custom layers (must match training notebook)
# --------------------------------------------------------------------------
class ECABlock(layers.Layer):
    """Efficient Channel Attention block."""

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.conv1d = layers.Conv1D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False
        )

    def call(self, inputs):
        x = self.avg_pool(inputs)                     # (B, C)
        x = tf.expand_dims(x, axis=-1)                 # (B, C, 1)
        x = self.conv1d(x)                              # (B, C, 1)
        x = tf.nn.sigmoid(x)
        x = tf.squeeze(x, axis=-1)                      # (B, C)
        x = tf.reshape(x, (-1, 1, 1, tf.shape(x)[-1]))   # (B,1,1,C)
        return inputs * x


class MSAM(layers.Layer):
    """Multi-Scale Attention Module: parallel 1x1 / 3x3 / 5x5 conv branches,
    each refined with an ECABlock, concatenated together."""

    def __init__(self, filters=256, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = layers.Conv2D(filters, kernel_size=1, padding='same', activation='relu')
        self.eca1 = ECABlock()

        self.conv3 = layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')
        self.eca3 = ECABlock()

        self.conv5 = layers.Conv2D(filters, kernel_size=5, padding='same', activation='relu')
        self.eca5 = ECABlock()

        self.concat = layers.Concatenate()

    def call(self, inputs):
        b1 = self.eca1(self.conv1(inputs))
        b2 = self.eca3(self.conv3(inputs))
        b3 = self.eca5(self.conv5(inputs))
        return self.concat([b1, b2, b3])


CUSTOM_OBJECTS = {
    "ECABlock": ECABlock,
    "MSAM": MSAM,
}


# --------------------------------------------------------------------------
# Grad-CAM model builders
# --------------------------------------------------------------------------
def build_gradcam_models(model):
    """
    Build two sub-models that expose intermediate activations for Grad-CAM:
      - grad_model_before: activations at the EfficientNetV2 backbone's last
        conv layer ("top_activation"), i.e. BEFORE the MSAM attention block.
      - grad_model_after: activations at the last MSAM layer in the model,
        i.e. AFTER the MSAM attention block.
    """
    before_layer_name = "top_activation"
    after_layer_name = None

    for layer in model.layers:
        if layer.__class__.__name__ == "MSAM":
            after_layer_name = layer.name  # keep the last MSAM layer found

    if after_layer_name is None:
        raise ValueError("No MSAM layer found in the loaded model.")

    grad_model_before = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(before_layer_name).output, model.output]
    )

    grad_model_after = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(after_layer_name).output, model.output]
    )

    return grad_model_before, grad_model_after


def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    """
    img_array: (1, 224, 224, 3) float32, raw pixel range 0-255
               (matches how the model was trained - no manual rescaling,
               EfficientNetV2 handles preprocessing internally).
    """
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy(), predictions.numpy()


def overlay_heatmap(img_0_1, heatmap, alpha=0.4):
    """
    img_0_1: (224, 224, 3) numpy array, RGB, values in [0, 1]
    heatmap: (h, w) numpy array, values in [0, 1] (Grad-CAM output)

    Returns an RGB uint8 overlay image ready to save/display.
    """
    heatmap = cv2.resize(heatmap, (img_0_1.shape[1], img_0_1.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)   # BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    base = np.uint8(np.clip(img_0_1, 0, 1) * 255)

    overlay = cv2.addWeighted(base, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def generate_gradcam_images(model, img_batch_0_255, img_0_1, pred_index=None):
    """
    Convenience wrapper: builds both grad-cam sub-models and returns the
    before/after MSAM overlay images (RGB uint8 numpy arrays).

    img_batch_0_255: (1, 224, 224, 3) float32, raw 0-255 pixel range
    img_0_1: (224, 224, 3) float32, pixel range 0-1 (for overlay background)
    """
    grad_model_before, grad_model_after = build_gradcam_models(model)

    heatmap_before, _ = make_gradcam_heatmap(img_batch_0_255, grad_model_before, pred_index)
    heatmap_after, _ = make_gradcam_heatmap(img_batch_0_255, grad_model_after, pred_index)

    overlay_before = overlay_heatmap(img_0_1, heatmap_before)
    overlay_after = overlay_heatmap(img_0_1, heatmap_after)

    return overlay_before, overlay_after

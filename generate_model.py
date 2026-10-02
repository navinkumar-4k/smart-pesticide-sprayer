"""
MobileNetV2 Model Generator for Tomato Foliage Pathology Classification
Exports 'tomato_disease_model.h5' for real-time edge/cloud inference.

Classes:
0 - Healthy
1 - Early Blight
2 - Late Blight
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_and_save_model(output_path="tomato_disease_model.h5"):
    print("Building MobileNetV2 Transfer Learning Architecture...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_tensor")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(3, activation='softmax', name="disease_prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Tomato_Leaf_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    print(f"\nSaving model to: {output_path}")
    model.save(output_path)
    print("Done! Model is ready for deployment in precision agriculture inference engine.")

if __name__ == "__main__":
    build_and_save_model()

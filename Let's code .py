import argparse
import logging
import os
import math
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Keras (modern multi-backend 3.x)
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0

save_dir = "D:/arcface_model"
os.makedirs(save_dir, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArcMarginProduct(tf.tensorflow.keras.layers.Layer):
    def __init__(self, n_classes, s=30.0, m=0.50, easy_margin=False, **kwargs):
        super(ArcMarginProduct, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = tf.constant(math.cos(m), dtype=tf.float32)
        self.sin_m = tf.constant(math.sin(m), dtype=tf.float32)
        self.th = tf.constant(math.cos(math.pi - m), dtype=tf.float32)
        self.mm = tf.constant(math.sin(math.pi - m) * m, dtype=tf.float32)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[0][-1], self.n_classes),
            initializer='he_normal',
            trainable=True
        )

    def call(self, inputs):
        x, label = inputs
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        cosine = tf.matmul(x, W)
        sine = tf.sqrt(1.0 - tf.square(cosine))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = tf.one_hot(label, depth=self.n_classes)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class SimCLRArcFacePipeline:
    def __init__(self, X, y_raw, num_classes=18):
        self.X = X
        self.y_raw = y_raw
        self.num_classes = num_classes
        self.simclr_model = self.create_simclr_projection_model()
        self.base_model = EfficientNetV2B0(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
        self.model = None
        self.prepare_data()

    def prepare_data(self):
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y_raw, test_size=0.3, stratify=self.y_raw)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        y_train_labels = np.argmax(y_train_cat, axis=1)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        self.class_weights = dict(enumerate(class_weights))

        self.X_train, self.y_train, self.y_val, self.y_test = X_train, y_train_cat, y_val_cat, y_test_cat
        self.y_train_int = np.argmax(self.y_train, axis=1)
        self.y_val_int = np.argmax(self.y_val, axis=1)
        self.y_test_int = np.argmax(self.y_test, axis=1)
        self.X_test_final = X_test

    def simclr_augment(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(image, size=[224, 224, 3])
        image = tf.image.random_brightness(image, 0.2)
        return image

    def create_simclr_projection_model(self):
        base = EfficientNetV2B0(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
        inputs = Input(shape=(224, 224, 3))
        features = base(inputs)
        projection = Dense(256, activation='relu')(features)
        projection = Dense(128)(projection)
        return Model(inputs, projection)

    def contrastive_loss(self, z1, z2, temperature=0.1):
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        representations = tf.concat([z1, z2], axis=0)
        similarity = tf.matmul(representations, representations, transpose_b=True)

        batch_size = tf.shape(z1)[0]
        labels = tf.range(batch_size)
        labels = tf.concat([labels + batch_size, labels], axis=0)
        labels = tf.cast(labels, tf.int32)

        logits = similarity / temperature
        loss = tf.tensorflow.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return tf.reduce_mean(loss)

    def simclr_train(self, steps=100):
        optimizer = tf.tensorflow.keras.optimizers.Adam(1e-3)
        for step in range(steps):
            idx = np.random.choice(len(self.X_train), 32)
            batch = tf.convert_to_tensor(self.X_train[idx], dtype=tf.float32)
            x1 = self.simclr_augment(batch)
            x2 = self.simclr_augment(batch)
            with tf.GradientTape() as tape:
                z1 = self.simclr_model(x1, training=True)
                z2 = self.simclr_model(x2, training=True)
                loss = self.contrastive_loss(z1, z2)
            grads = tape.gradient(loss, self.simclr_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.simclr_model.trainable_variables))
            if step % 10 == 0:
                logging.info(f"SimCLR Step {step}: Loss = {loss.numpy():.4f}")

    def build_model(self):
        self.base_model.trainable = False
        inputs = Input(shape=(224, 224, 3))
        x = preprocess_input(inputs)
        features = self.base_model(x, training=False)
        features = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(features)
        features = BatchNormalization()(features)
        features = Dropout(0.5)(features)
        features = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(features)
        features = BatchNormalization()(features)
        features = Dropout(0.3)(features)
        label_input = Input(shape=(), dtype='int32')
        outputs = ArcMarginProduct(n_classes=self.num_classes)([features, label_input])
        self.model = Model(inputs=[inputs, label_input], outputs=outputs)

    def train_model(self):
        self.model.compile(optimizer=Adam(1e-4), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
        ]
        self.model.fit([self.X_train, self.y_train_int], self.y_train_int,
                       validation_data=([self.X_test_final, self.y_val_int], self.y_val_int),
                       epochs=30,
                       class_weight=self.class_weights,
                       callbacks=callbacks)
        save_dir = "D:/arcface_model"
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_weights(os.path.join(save_dir, "arcface_model_weights.h5"))
        logging.info(f"Model weights saved to {save_dir}/arcface_model_weights.h5")
        self.model.save(os.path.join(save_dir, "arcface_full_model.tensorflow.keras"))
        logging.info(f"Full model saved to {save_dir}/arcface_full_model.tensorflow.keras")



    def fine_tune_model(self):
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-50]:
            layer.trainable = False
        self.model.compile(optimizer=Adam(1e-5), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        self.model.fit([self.X_train, self.y_train_int], self.y_train_int,
                       validation_data=([self.X_test_final, self.y_val_int], self.y_val_int),
                       epochs=20,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
        save_dir = "D:/arcface_model"
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_weights(os.path.join(save_dir, "arcface_model_weights.h5"))
        logging.info(f"Model weights saved to {save_dir}/arcface_model_weights.h5")
        self.model.save(os.path.join(save_dir, "arcface_full_model.tensorflow.keras"))
        logging.info(f"Full model saved to {save_dir}/arcface_full_model.tensorflow.keras")

    def evaluate_model(self):
        y_pred = self.model.predict([self.X_test_final, self.y_test_int])
        acc = self.model.evaluate([self.X_test_final, self.y_test_int], self.y_test_int)
        logging.info(f"Final Test Accuracy: {acc}")
        print(classification_report(self.y_test_int, np.argmax(y_pred, axis=1)))

    def run_pipeline(self):
        logging.info("Starting SimCLR Pretraining...")
        self.simclr_train(steps=100)
        logging.info("Building ArcFace Classification Model...")
        self.build_model()
        logging.info("Training Phase 1...")
        self.train_model()
        logging.info("Fine-Tuning Phase 2...")
        self.fine_tune_model()
        logging.info("Evaluating Final Model...")
        self.evaluate_model()

def main():
    # Setup argument parser
    print("✅ Keras backend is:", tensorflow.keras.backend.backend())

    parser = argparse.ArgumentParser(description="Train a model using SimCLR + ArcFace pipeline")
    parser.add_argument('--data_path', type=str, default='dataset.npz',
                        help='Path to the .npz file containing X and y_raw arrays')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of SimCLR pretraining steps')
    args = parser.parse_args()

    # Load data (example: from a .npz file)
    logging.info("Loading dataset...")
    try:
        data = np.load(args.data_path)
        X = data['X']
        y_raw = data['y']
        logging.info(f"Loaded data: X shape {X.shape}, y shape {y_raw.shape}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Run pipeline
    logging.info("Initializing training pipeline...")
    pipeline = SimCLRArcFacePipeline(X, y_raw)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()

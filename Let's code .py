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

# TensorFlow Keras imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Multiply, Concatenate, Conv2D
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, RandomRotation, RandomTranslation, RandomZoom
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create directory to save models
save_dir = "D:/arcface_model"
os.makedirs(save_dir, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArcMarginProduct(tf.keras.layers.Layer):
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

        self.X_train, self.y_train, self.y_val, self.y_test = X_train, y_train_cat, y_val_cat, y_test_cat
        self.y_train_int = np.argmax(y_train_cat, axis=1)
        self.y_val_int = np.argmax(y_val_cat, axis=1)
        self.y_test_int = np.argmax(y_test_cat, axis=1)
        self.X_test_final = X_test

        # Compute class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train_int), y=self.y_train_int)
        self.class_weights = dict(enumerate(class_weights))

    def simclr_augment(self, image):
        image = tf.image.random_flip_left_right(image)
        image = RandomRotation(0.07)(image, training=True)
        image = RandomTranslation(0.15, 0.15)(image, training=True)
        image = RandomZoom((-0.3, 0.3), (-0.3, 0.3))(image, training=True)
        image = tf.image.random_brightness(image, 0.2)
        return tf.image.resize_with_crop_or_pad(image, 224, 224)

    def create_simclr_projection_model(self):
        base = EfficientNetV2B0(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
        inputs = Input(shape=(224, 224, 3))
        x = base(inputs)
        x = Dense(256, activation='relu')(x)
        projection = Dense(128)(x)
        return Model(inputs, projection)

    def supervised_contrastive_loss(self, z, labels, temperature=0.1):
        z = tf.math.l2_normalize(z, axis=1)
        similarity = tf.matmul(z, z, transpose_b=True) / temperature
        labels = tf.expand_dims(labels, 1)
        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
        logits_mask = 1.0 - tf.eye(tf.shape(labels)[0])
        exp_sim = tf.exp(similarity) * logits_mask
        log_prob = similarity - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-9)
        loss = -tf.reduce_sum(mask * log_prob, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-9)
        return tf.reduce_mean(loss)

    def simclr_train(self, steps=100):
        optimizer = tf.keras.optimizers.Adam(1e-3)
        for step in range(steps):
            idx = np.random.choice(len(self.X_train), 32)
            batch = tf.convert_to_tensor(self.X_train[idx], dtype=tf.float32)
            labels = tf.convert_to_tensor(self.y_train_int[idx], dtype=tf.int32)

            x1 = self.simclr_augment(batch)
            x2 = self.simclr_augment(batch)

            with tf.GradientTape() as tape:
                z1 = self.simclr_model(x1, training=True)
                z2 = self.simclr_model(x2, training=True)
                z = tf.concat([z1, z2], axis=0)
                y = tf.concat([labels, labels], axis=0)
                loss = self.supervised_contrastive_loss(z, y)

            grads = tape.gradient(loss, self.simclr_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.simclr_model.trainable_variables))

            if step % 10 == 0:
                logging.info(f"SimCLR SupCon Step {step}: Loss = {loss.numpy():.4f}")

    def cbam_block(self, input_tensor, reduction_ratio=8):
        channels = input_tensor.shape[-1]
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        max_pool = GlobalMaxPooling2D()(input_tensor)
        shared_dense = tf.keras.Sequential([
            Dense(channels // reduction_ratio, activation='relu'),
            Dense(channels)
        ])
        channel_attention = tf.keras.activations.sigmoid(shared_dense(avg_pool) + shared_dense(max_pool))
        channel_attention = Reshape((1, 1, channels))(channel_attention)
        x = Multiply()([input_tensor, channel_attention])

        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        spatial_attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
        x = Multiply()([x, spatial_attention])
        return x

    def mixup(self, x, y, alpha=0.2):
        batch_size = tf.shape(x)[0]
        lam = np.random.beta(alpha, alpha)
        index = tf.random.shuffle(tf.range(batch_size))
        x_mix = lam * x + (1 - lam) * tf.gather(x, index)
        y_mix = lam * y + (1 - lam) * tf.gather(y, index)
        return x_mix, y_mix

    def build_model(self):
        # Freeze base model initially
        self.base_model.trainable = False

        # Inputs
        image_input = Input(shape=(224, 224, 3), name='image_input')
        label_input = Input(shape=(), dtype='int32', name='label_input')

        # Preprocess input for EfficientNet
        x = preprocess_input(image_input)

        # Base CNN features
        features = self.base_model(x, training=False)

        # ✅ CBAM Attention Block
        features = self.cbam_block(features)

        # Fully connected head
        features = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(features)
        features = BatchNormalization()(features)
        features = Dropout(0.5)(features)

        features = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(features)
        features = BatchNormalization()(features)
        features = Dropout(0.3)(features)

        # ArcFace final classification head
        arc_output = ArcMarginProduct(n_classes=self.num_classes)([features, label_input])

        # Final model with two inputs (image, label), one output (logits for classification)
        self.model = Model(inputs=[image_input, label_input], outputs=arc_output)


    def train_model(self):
        logging.info("Starting Supervised ArcFace Training with MixUp...")

        # Apply MixUp on the full training set
        x_mix, y_mix = self.mixup(self.X_train, self.y_train)

        # Use integer labels for ArcFace (not one-hot)
        y_mix_int = np.argmax(y_mix, axis=1)

        # Cosine learning rate scheduler
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-4,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-5
        )

        optimizer = Adam(learning_rate=lr_schedule)

        self.model.compile(
            optimizer=optimizer,
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            # Optional if you want dynamic LR reduction based on plateaus
            # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]

        # Fit the model
        self.model.fit(
            [x_mix, y_mix_int],  # [images, labels]
            y_mix_int,           # ArcFace expects sparse int labels
            validation_data=([self.X_test_final, self.y_val_int], self.y_val_int),
            epochs=30,
            class_weight=self.class_weights,
            callbacks=callbacks
        )

        # Save checkpoints
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_weights(os.path.join(save_dir, "arcface_model_weights.h5"))
        self.model.save(os.path.join(save_dir, "arcface_full_model.tensorflow.keras"))
        logging.info(f"✅ Model saved to: {save_dir}")

    def fine_tune_model(self):
        logging.info("🔁 Starting Fine-Tuning Phase 1: Unfreezing last 50 layers...")

        # Phase 1: Unfreeze last 50 layers
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-50]:
            layer.trainable = False
        for layer in self.base_model.layers[-50:]:
            layer.trainable = True

        lr_schedule_phase1 = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-5,
            first_decay_steps=500,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-6
        )

        self.model.compile(
            optimizer=Adam(learning_rate=lr_schedule_phase1),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.model.fit(
            [self.X_train, self.y_train_int], self.y_train_int,
            validation_data=([self.X_test_final, self.y_val_int], self.y_val_int),
            epochs=10,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )

        logging.info("🧠 Starting Fine-Tuning Phase 2: Unfreezing all layers...")

        # Phase 2: Unfreeze all layers
        for layer in self.base_model.layers:
            layer.trainable = True

        lr_schedule_phase2 = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=5e-6,
            first_decay_steps=500,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-6
        )

        self.model.compile(
            optimizer=Adam(learning_rate=lr_schedule_phase2),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.model.fit(
            [self.X_train, self.y_train_int], self.y_train_int,
            validation_data=([self.X_test_final, self.y_val_int], self.y_val_int),
            epochs=10,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )

        # Save final fine-tuned model
        self.model.save_weights(os.path.join(save_dir, "arcface_model_weights.h5"))
        self.model.save(os.path.join(save_dir, "arcface_full_model.tensorflow.keras"))
        logging.info(f"✅ Final fine-tuned model saved to: {save_dir}")

    def evaluate_model(self):
        logging.info("📊 Evaluating final model on test set...")

        # Get logits from ArcFace head
        y_pred_logits = self.model.predict([self.X_test_final, self.y_test_int])
        
        # Convert logits to predicted labels
        y_pred_labels = np.argmax(y_pred_logits, axis=1)

        # Evaluate accuracy
        test_acc = self.model.evaluate([self.X_test_final, self.y_test_int], self.y_test_int, verbose=0)
        logging.info(f"✅ Final Test Accuracy: {test_acc[1] * 100:.2f}%")

        # Classification report
        print("\n🧾 Classification Report:\n")
        print(classification_report(self.y_test_int, y_pred_labels, digits=4))

    def run_pipeline(self):
        logging.info("🚀 Starting SimCLR Pretraining (Supervised Contrastive Learning)...")
        self.simclr_train(steps=100)

        logging.info("🏗️ Building ArcFace Classification Model with CBAM Attention...")
        self.build_model()

        logging.info("📚 Supervised Training Phase (with MixUp)...")
        self.train_model()

        logging.info("🧠 Fine-Tuning Full Network...")
        self.fine_tune_model()

        logging.info("📊 Final Evaluation on Test Set...")
        self.evaluate_model()

        logging.info("✅ Full training and evaluation pipeline completed.")

def main():
    print("✅ Keras backend is:", tf.keras.backend.backend())
    print("🧠 Available GPUs:", tf.config.list_physical_devices('GPU'))

    parser = argparse.ArgumentParser(description="Train a model using SimCLR + ArcFace + CBAM + SupCon + MixUp pipeline")
    parser.add_argument('--steps', type=int, default=100, help='Number of SimCLR pretraining steps')
    args = parser.parse_args()

    # ✅ Load CIFAR-100 from Keras API
    logging.info("📂 Loading CIFAR-100 dataset (fine labels)...")
    from tensorflow.keras.datasets import cifar100
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

    # Flatten y arrays
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Normalize and resize to (224, 224)
    logging.info("📏 Resizing CIFAR-100 images to 224x224 for EfficientNetV2...")
    X_train = tf.image.resize(tf.cast(X_train, tf.float32), [224, 224]) / 255.0
    X_test  = tf.image.resize(tf.cast(X_test, tf.float32), [224, 224])/ 255.0

    # Combine for unified preprocessing
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    logging.info(f"✅ Loaded CIFAR-100: X shape = {X.shape}, y shape = {y.shape}, num_classes = 100")

    # Initialize and run pipeline
    pipeline = SimCLRArcFacePipeline(X, y, num_classes=100)
    pipeline.run_pipeline()

    logging.info("🏁 Training complete.")


if __name__ == "__main__":
    main()

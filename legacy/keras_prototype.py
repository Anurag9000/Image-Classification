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
from tensorflow.keras.datasets import cifar100


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import keras_tuner as kt

# Create directory to save models
save_dir = "./arcface_model"
os.makedirs(save_dir, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hyperparameters batch size definition define
BATCH_SIZE = 32

def build_tuned_model(hp):
    arcface_m = hp.Float('arcface_margin', min_value=0.2, max_value=0.7, step=0.05)
    arcface_s = hp.Choice('arcface_scale', values=[10.0, 30.0, 64.0])

    dropout_rate = hp.Float('dropout', 0.3, 0.7, step=0.1)
    dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=64)

    base_model = EfficientNetV2B0(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
    base_model.trainable = False

    image_input = Input(shape=(224, 224, 3), name='image_input')
    label_input = Input(shape=(), dtype='int32', name='label_input')

    x = preprocess_input(image_input)
    features = base_model(x)
    features = Dense(dense_units, activation='relu')(features)
    features = Dropout(dropout_rate)(features)
    
    arc_output = ArcMarginProduct(n_classes=100, m=arcface_m, s=arcface_s)([features, label_input])
    
    model = Model(inputs=[image_input, label_input], outputs=arc_output)
    model.compile(
        optimizer=Adam(hp.Choice('lr', values=[1e-3, 1e-4, 1e-5])),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def resize_generator(images, labels, size=(224, 224), batch_size=BATCH_SIZE):
    for i in range(0, len(images), batch_size):
        batch_images = tf.image.resize(tf.cast(images[i:i+batch_size], tf.float32), size) / 255.0
        batch_labels = labels[i:i+batch_size]
        yield batch_images.numpy(), batch_labels


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
    def __init__(self, train_ds, val_ds, test_ds, num_classes=18):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.num_classes = num_classes
        self.simclr_model = self.create_simclr_projection_model()
        self.base_model = EfficientNetV2B0(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
        self.model = None

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
        for step, (batch_images, batch_labels) in enumerate(self.train_ds.repeat()):
            if step >= steps:
                break
            batch = tf.convert_to_tensor(batch_images, dtype=tf.float32)
            labels = tf.convert_to_tensor(batch_labels, dtype=tf.int32)

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
    
    def cutmix(self, x, y, alpha=1.0):
        batch_size = tf.shape(x)[0]
        lam = np.random.beta(alpha, alpha)
        index = tf.random.shuffle(tf.range(batch_size))
        x1 = tf.gather(x, index)
        y1 = tf.gather(y, index)
        rx = tf.random.uniform([], 0, 224, dtype=tf.int32)
        ry = tf.random.uniform([], 0, 224, dtype=tf.int32)
        rw = tf.random.uniform([], 0, 224 // 2, dtype=tf.int32)
        rh = tf.random.uniform([], 0, 224 // 2, dtype=tf.int32)
        x_cut = tf.identity(x)
        x_cut[:, rx:rx+rw, ry:ry+rh, :] = x1[:, rx:rx+rw, ry:ry+rh, :]
        lam_adjusted = 1 - (rw * rh) / (224 * 224)
        y_mix = lam_adjusted * y + (1 - lam_adjusted) * y1
        return x_cut, y_mix
    
    def build_model(self):
        self.base_model.trainable = False

        image_input = Input(shape=(224, 224, 3), name='image_input')
        label_input = Input(shape=(), dtype='int32', name='label_input')

        x = preprocess_input(image_input)
        features = self.base_model(x, training=False)
        features = self.cbam_block(features)

        features = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(features)
        features = BatchNormalization()(features)
        features = Dropout(0.5)(features)

        features = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(features)
        features = BatchNormalization()(features)
        features = Dropout(0.3)(features)

        arc_output = ArcMarginProduct(n_classes=self.num_classes)([features, label_input])
        self.model = Model(inputs=[image_input, label_input], outputs=arc_output)

    def train_model(self):
        logging.info("Starting Supervised ArcFace Training...")

        def mixup_batch(batch):
            images, labels = batch
            labels_one_hot = tf.one_hot(labels, depth=self.num_classes)

            # Choose MixUp or CutMix randomly
            if tf.random.uniform([]) > 0.5:
                mixed_images, mixed_labels = self.mixup(images, labels_one_hot)
            else:
                mixed_images, mixed_labels = self.cutmix(images, labels_one_hot)

            return (mixed_images, tf.argmax(mixed_labels, axis=1)), tf.argmax(mixed_labels, axis=1)


        mixed_train_ds = self.train_ds.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)
        mixed_train_ds = mixed_train_ds.prefetch(tf.data.AUTOTUNE)

        val_ds = self.val_ds.map(lambda x, y: ((x, y), y)).prefetch(tf.data.AUTOTUNE)

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

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ]

        self.model.fit(
            mixed_train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=callbacks,
            verbose=1
        )

        self.model.save_weights(os.path.join(save_dir, "arcface_model_weights.h5"))
        self.model.save(os.path.join(save_dir, "arcface_full_model.tensorflow.keras"))
        logging.info(f"✅ Model saved to: {save_dir}")

    def fine_tune_model(self):
        logging.info("🔁 Fine-Tuning Phase 1: Unfreezing last 50 layers...")

        self.base_model.trainable = True
        for layer in self.base_model.layers[:-50]:
            layer.trainable = False

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

        val_ds = self.val_ds.map(lambda x, y: ((x, y), y)).prefetch(tf.data.AUTOTUNE)
        train_ds = self.train_ds.map(lambda x, y: ((x, y), y)).prefetch(tf.data.AUTOTUNE)

        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=1
        )

        logging.info("🧠 Fine-Tuning Phase 2: Unfreezing all layers...")

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
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=1
        )

        self.model.save_weights(os.path.join(save_dir, "arcface_model_weights.h5"))
        self.model.save(os.path.join(save_dir, "arcface_full_model.tensorflow.keras"))
        logging.info(f"✅ Final fine-tuned model saved to: {save_dir}")

    def evaluate_model(self):
        logging.info("📊 Evaluating final model on test set...")

        y_true, y_pred = [], []
        for (x, y) in self.test_ds.unbatch().take(1000):  # or full
            pred = self.model.predict([[tf.expand_dims(x, 0)], [tf.expand_dims(y, 0)]], verbose=0)
            y_true.append(y.numpy())
            y_pred.append(np.argmax(pred))

        # ✅ Classification report
        print("\n📈 Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        # ✅ Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(14, 10))
        sns.heatmap(cm, cmap='Blues', xticklabels=range(self.num_classes), yticklabels=range(self.num_classes), annot=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()


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

def compute_gradcam(model, image, label_index, conv_layer_name='efficientnetv2-b0'):
    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[model.get_layer(conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([tf.expand_dims(image, 0), tf.expand_dims(label_index, 0)])
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = tf.image.resize(tf.expand_dims(cam, -1), (224, 224)).numpy()
    cam = cam / (np.max(cam) + 1e-8)
    return cam


def preprocess(image, label):
    image = tf.image.resize(tf.cast(image, tf.float32), [224, 224]) / 255.0
    label = tf.cast(label, tf.int32)  # ✅ Cast to int32
    return image, label


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("✅ Keras backend is:", tf.keras.backend.backend())
    print("🧠 Available GPUs:", tf.config.list_physical_devices('GPU'))

    # tuner = kt.Hyperband(
    # build_tuned_model,
    # objective='val_accuracy',
    # max_epochs=10,
    # factor=3,
    # directory='kerastuner_dir',
    # project_name='arcface_cifar100'
    # )

    # val_ds = val_ds.map(lambda x, y: ((x, y), y)).prefetch(tf.data.AUTOTUNE)
    # train_ds = train_ds.map(lambda x, y: ((x, y), y)).prefetch(tf.data.AUTOTUNE)

    # tuner.search(train_ds, validation_data=val_ds, epochs=10)
    # best_model = tuner.get_best_models(num_models=1)[0]


    # parser = argparse.ArgumentParser(description="Train a model using SimCLR + ArcFace + CBAM + SupCon + MixUp pipeline")
    # parser.add_argument('--steps', type=int, default=100, help='Number of SimCLR pretraining steps')
    # args = parser.parse_args()

    # ✅ Load CIFAR-100
    logging.info("📂 Loading CIFAR-100 dataset (fine labels)...")
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Combine and create full dataset
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    total_samples = len(X_all)

    dataset = tf.data.Dataset.from_tensor_slices((X_all, y_all))
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # ✅ Split: 70% train, 15% val, 15% test
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size

    train_ds = dataset.take(train_size // BATCH_SIZE)
    val_ds = dataset.skip(train_size // BATCH_SIZE).take(val_size // BATCH_SIZE)
    test_ds = dataset.skip((train_size + val_size) // BATCH_SIZE)

    logging.info("✅ Dataset split into Train / Val / Test")

    # ✅ Run full training pipeline
    pipeline = SimCLRArcFacePipeline(train_ds, val_ds, test_ds, num_classes=100)
    pipeline.run_pipeline()

    logging.info("🏁 Training complete.")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import tempfile
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight

# Path to the datasets
datasets_path = r"c:\Users\gento\Desktop\Projects\RBG IITM\Datasets"

# List of dataset files
dataset_files = [
    "normal_posture_cleaned.csv",
    "pelvis_rearward_rotation_cleaned.csv",
    "reclining_back_cleaned.csv",
    "rounded_back_cleaned.csv",
    "thorax_forward_rotation_cleaned.csv",
    "thorax_rearward_rotation_cleaned.csv"
]

# Load and prepare the data
def load_data():
    X_all = []
    y_all = []
    
    print("Loading datasets:")
    for file in dataset_files:
        # Extract class name from file name
        class_name = file.replace("_cleaned.csv", "")
        
        # Load the dataset
        file_path = os.path.join(datasets_path, file)
        df = pd.read_csv(file_path)
        
        # Add the data to our lists
        X_all.append(df)
        # Create labels for each row in this dataset
        y_all.append([class_name] * len(df))
        
        print(f"  {file}: {len(df)} samples")
    
    # Concatenate all data
    X = pd.concat(X_all, axis=0, ignore_index=True)
    y = np.concatenate(y_all)
    
    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale the features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save the scaler for inference
    scaler_path = os.path.join(datasets_path, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Feature scaler saved to: {scaler_path}")
    print(f"Total dataset size: {len(X)} samples")
    
    # Check class balance
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  {cls}: {count} samples ({count/len(y):.2%})")
    
    return X_scaled_df, y_encoded, label_encoder

# Train XGBoost model with cross-validation
def train_xgboost_model(X, y):
    # Set parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'max_depth': 4,  # Reduced from 6 to prevent overfitting
        'eta': 0.1,      # Reduced from 0.3 to prevent overfitting
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,  # L1 regularization to prevent overfitting
        'reg_lambda': 1.0   # L2 regularization to prevent overfitting
    }
    
    # Perform k-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    print("Performing 5-fold cross-validation for XGBoost model:")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train the model
        num_rounds = 100
        model = xgb.train(
            params, 
            dtrain, 
            num_rounds, 
            evals=[(dval, 'validation')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Evaluate the model
        preds = model.predict(dval)
        best_preds = np.asarray([np.argmax(line) for line in preds])
        accuracy = accuracy_score(y_val, best_preds)
        cv_scores.append(accuracy)
        
        print(f"  Fold {fold+1}: Accuracy = {accuracy:.4f}")
    
    print(f"Cross-validation mean accuracy: {np.mean(cv_scores):.4f}, std: {np.std(cv_scores):.4f}")
    
    # Train final model on all data
    dtrain_full = xgb.DMatrix(X, label=y)
    final_model = xgb.train(params, dtrain_full, num_rounds)
    
    return final_model

# Create a TensorFlow model that can be converted to TFLite
def create_tf_model(num_features, num_classes):
    """Create a TensorFlow model for classification with regularization"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_features,), name='input'),
        tf.keras.layers.BatchNormalization(),  # Add batch normalization
        tf.keras.layers.Dense(128, activation='relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),  # Increased dropout rate
        tf.keras.layers.Dense(64, activation='relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train TensorFlow model with cross-validation
def train_tf_model(X, y, num_classes):
    """Train a TensorFlow model with cross-validation"""
    # Convert to numpy arrays
    X_np = X.values
    
    # Calculate class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("Class weights to handle imbalance:")
    for cls, weight in class_weights_dict.items():
        print(f"  Class {cls}: {weight:.4f}")
    
    # Perform k-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    cv_reports = []
    
    print("Performing 5-fold cross-validation for TensorFlow model:")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_np, y)):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create and train model
        model = create_tf_model(X.shape[1], num_classes)
        
        # Use early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Train the model
        history = model.fit(
            X_train, 
            y_train,
            epochs=100,  # Increased epochs
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights_dict,  # Use class weights
            verbose=0
        )
        
        # Evaluate the model
        y_pred = np.argmax(model.predict(X_val), axis=1)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        cv_scores.append(accuracy)
        cv_reports.append(report)
        
        print(f"  Fold {fold+1}: Accuracy = {accuracy:.4f}")
        
        # Print per-class metrics for this fold
        print(f"    Per-class metrics for fold {fold+1}:")
        for cls in range(num_classes):
            precision = report[str(cls)]['precision']
            recall = report[str(cls)]['recall']
            f1 = report[str(cls)]['f1-score']
            print(f"      Class {cls}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    print(f"Cross-validation mean accuracy: {np.mean(cv_scores):.4f}, std: {np.std(cv_scores):.4f}")
    
    # Train final model on all data
    final_model = create_tf_model(X.shape[1], num_classes)
    
    # Use early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )
    
    # Train the model
    final_model.fit(
        X_np, 
        y,
        epochs=100,  # Increased epochs
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_dict,  # Use class weights
        verbose=1
    )
    
    # Evaluate final model on training data to check for overfitting
    y_pred = np.argmax(final_model.predict(X_np), axis=1)
    print("\nFinal model performance on training data:")
    print(classification_report(y, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(datasets_path, 'training_confusion_matrix.png'))
    plt.close()
    print("Confusion matrix saved to 'training_confusion_matrix.png'")
    
    return final_model

# Convert TensorFlow model to TFLite
def convert_to_tflite(tf_model, label_encoder):
    """Convert TensorFlow model to TFLite format"""
    # Save the TensorFlow model
    tf_model_path = os.path.join(datasets_path, "tf_model")
    tf_model.save(tf_model_path)
    
    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()
    
    # Save the TFLite model
    tflite_path = os.path.join(datasets_path, "posture_classification_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {tflite_path}")
    
    # Save label encoder classes for later use
    classes = label_encoder.classes_
    classes_path = os.path.join(datasets_path, "posture_classes.txt")
    with open(classes_path, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    print(f"Class labels saved to: {classes_path}")
    
    return tflite_path

def main():
    print("Loading and preparing data...")
    X, y, label_encoder = load_data()
    
    print("\nTraining XGBoost model with cross-validation...")
    xgb_model = train_xgboost_model(X, y)
    
    # Save the XGBoost model for reference
    xgb_model_path = os.path.join(datasets_path, "xgboost_model.pkl")
    with open(xgb_model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    
    print(f"Original XGBoost model saved to: {xgb_model_path}")
    
    print("\nTraining TensorFlow model with cross-validation...")
    num_classes = len(np.unique(y))
    tf_model = train_tf_model(X, y, num_classes)
    
    print("\nConverting model to TFLite...")
    tflite_path = convert_to_tflite(tf_model, label_encoder)
    
    print("\nDone!")
    print(f"Model saved to: {tflite_path}")

if __name__ == "__main__":
    main()

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix


def count_labels(y_tensor):
  # Cuenta cuántas veces aparece cada etiqueta
  labels = y_tensor.tolist()
  label_counts = Counter(labels)
  print(label_counts)

def create_segment_label(value, N):
    """
    Create an array of length N filled with the given label value.
    """
    return np.ones(N) * value

def generate_datasets(X_train_C, X_test_C, y_train_C, y_test_C):
    """
    Generate segment-level datasets from trial-level data.

    For each trial, replicate its label for all its segments,
    then stack all segments and labels into final arrays.
    """
    y_train = []
    y_test = []

    # Generate labels for each segment in training data
    for idx, y in enumerate(y_train_C):
        N = X_train_C[idx].shape[0]
        y_train.append(create_segment_label(y, N))

    # Generate labels for each segment in test data
    for idx, y in enumerate(y_test_C):
        N = X_test_C[idx].shape[0]
        y_test.append(create_segment_label(y, N))

    # Concatenate all labels and feature segments
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    X_train = np.vstack(X_train_C)
    X_test = np.vstack(X_test_C)

    return X_train, X_test, y_train, y_test

def extract_identifier(filename, prefix):
    """Extracts the identifier (N1_C0) from a filename based on a given prefix."""
    pattern = rf"{prefix}_(N\d+_C\d+)\."
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def data_preprocess(data_dir, labels_dir):
    """Loads and processes EEG data and labels, ensuring alignment and preserving label column names."""

    # Get sorted EEG and label files
    eeg_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".feather")])

    # Create dictionaries mapping identifier -> filename
    eeg_dict = {extract_identifier(f, "EEG"): f for f in eeg_files}
    label_dict = {extract_identifier(f, "labels"): f for f in label_files}

    # Extract identifiers from filenames
    eeg_ids = set(eeg_dict.keys())
    label_ids = set(label_dict.keys())

    # Find missing labels and EEG files
    missing_labels = eeg_ids - label_ids  # EEG files without corresponding labels
    missing_eeg = label_ids - eeg_ids  # Labels without corresponding EEG files

    # Print missing files
    print(f"Total EEG files: {len(eeg_ids)}")
    print(f"Total label files: {len(label_ids)}")
    print(f"Missing label files: {len(missing_labels)}")
    print(f"Missing EEG files: {len(missing_eeg)}")
    print("EEG files without labels:", sorted(missing_labels))
    print("Label files without EEG data:", sorted(missing_eeg))

    # Ensure identifiers match for loading
    common_ids = sorted(eeg_ids & label_ids)

    # Load and concatenate EEG data
    eeg_data = [np.load(os.path.join(data_dir, eeg_dict[i])) for i in common_ids]
    eeg_data_array = np.concatenate(eeg_data, axis=0)

    # Load labels as DataFrames and concatenate
    label_dfs = [pd.read_feather(os.path.join(labels_dir, label_dict[i])) for i in common_ids]
    labels_df = pd.concat(label_dfs, axis=0, ignore_index=True)  # Keep column names

    # Print final shapes
    print("EEG Data Shape:", eeg_data_array.shape)
    print("Labels Shape:", labels_df.shape)

    return eeg_data_array, labels_df

def histogram_labels(labels_df, name):
    labels = labels_df[name].to_numpy()

    """Plots a histogram of label values."""
    plt.figure(figsize=(10, 5))
    plt.hist(labels, bins=50, edgecolor='black')
    plt.title(f'Histograma de {name}')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')

def plot_cm(conf_matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Etiquetas de clase
    class_names = []
    for i in range(classes):
        class_names.append('Clase ' + str(i))

    # Crear la figura con seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()


    # Normalizar por fila (eje=1)
    conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

    # Crear la figura con seaborn heatmap (valores normalizados)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión Normalizada por Fila')
    plt.tight_layout()
    plt.show()


def extract_class(eeg_data_array, labels_df):
    # Extraer valores continuos
    valence = labels_df['Valence'].to_numpy()
    arousal = labels_df['Arousal'].to_numpy()

    upper_limit_v = 6.5
    under_limit_v = 3.5

    upper_limit_a = 6.5
    under_limit_a = 3.5

    upper_limit_nv = 6.5
    under_limit_nv = 3.5

    upper_limit_na = upper_limit_a
    under_limit_na = under_limit_a

    # Definir condiciones para los 4 cuadrantes + clase neutra
    conditions_choices = [
        ((valence >= upper_limit_v) & (arousal >= upper_limit_a), 0),  # Alto V, Alto A
        ((valence < under_limit_v) & (arousal >= upper_limit_a), 1),   # Bajo V, Alto A
        ((valence < under_limit_v) & (arousal < under_limit_a), 2),    # Bajo V, Bajo A
        ((valence >= upper_limit_v) & (arousal < under_limit_a), 3),   # Alto V, Bajo A
        (
            (valence >= under_limit_nv) & (valence < upper_limit_nv) &
            (arousal >= under_limit_na) & (arousal < upper_limit_na),
            4  # Clase neutra (zona gris)
        )
    ]


    # Crear máscara general de validez (sólo se toman las muestras que caen en algún cuadrante)
    valid_mask = np.any([cond for cond, _ in conditions_choices], axis=0)

    # Filtrar datos
    filtered_eeg = eeg_data_array[valid_mask]

    # Etiquetas discretas por cuadrante
    discrete_labels = np.select(
        [cond[valid_mask] for cond, _ in conditions_choices],
        [label for _, label in conditions_choices]
    )

    return filtered_eeg, discrete_labels
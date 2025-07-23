
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import data_preprocess, generate_datasets, count_labels, extract_class
from objective import optuna_objective
import optuna
import argparse

def main(data_dir, labels_dir, results_dir):
    #Preprocessing data and generating classes
    eeg_data_array, labels_df = data_preprocess(data_dir, labels_dir)
    eeg_data_array, labels_df = extract_class(eeg_data_array, labels_df)

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        eeg_data_array, labels_df, test_size=0.2, random_state=42, stratify=labels_df
    )

    X_train, X_test, y_train, y_test = generate_datasets(X_train, X_test, y_train, y_test)

    print('Dataset dimensions:')
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    #Count labels
    count_labels(y_train)
    count_labels(y_test)


    # ----------- Lanzar la optimización -----------
    study = optuna.create_study(
        direction="maximize",
        study_name="MLP_DE_escolares",
        #storage= f"sqlite:///{results_dir}/optuna_study_all_VA_5class_f1macro.db",
        #load_if_exists=True  # evita sobreescribir si ya existe
    )
    opt_model = optuna_objective(X_train, X_test, y_train, y_test, n_classes=5, sample_weights_loss=True)
    study.optimize(opt_model.objective, n_trials=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EEG classification optimization with Optuna.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to EEG features directory")
    parser.add_argument("--labels_dir", type=str, required=True, help="Path to labels directory")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to store optimization results")

    args = parser.parse_args()
    main(args.data_dir, args.labels_dir, args.results_dir)

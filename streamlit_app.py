import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define all your functions and classes here

def fill_if_null(data):
    null_boy = np.array(data.columns[data.isnull().any()])
    for i in null_boy:
        data[i] = data[i].fillna(data[i].mean())
    return data

def check_null(data):
    st.write("Checking for null values in the data...")
    if data.isnull().values.any():
        fill_if_null(data)
        st.success("Null values found and filled with column mean.")
        st.write(data.isnull().sum())
    else:
        st.success("No null values found.")
        st.write(data.isnull().sum())

def XandY(data, dept):
    Y = data[dept].to_numpy()
    data.drop(dept, axis=1, inplace=True)
    X = data.to_numpy()
    return [X, Y]

class GradientBoostingTree:
    # Your GradientBoostingTree class code here
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, loss="squared_error"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.trees = []
        self.initial_prediction = None

    def initialize_model_parameters(self, y):
        return np.mean(y)

    def loss_gradient(self, y, pred):
        return y - pred  

    def fit_ensemble_tree(self, X, residuals):
        tree = self.construct_tree(X, residuals, depth=0)
        return tree

    def construct_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {"value": np.mean(y)}

        n_samples, n_features = X.shape
        best_split = None
        min_error = float("inf")

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                left_mean = np.mean(y[left_mask]) if np.any(left_mask) else 0
                right_mean = np.mean(y[right_mask]) if np.any(right_mask) else 0

                error = np.sum((y[left_mask] - left_mean) ** 2) + np.sum((y[right_mask] - right_mean) ** 2)

                if error < min_error:
                    min_error = error
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left_mean": left_mean,
                        "right_mean": right_mean,
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                    }

        left_tree = self.construct_tree(X[best_split["left_mask"]], y[best_split["left_mask"]], depth + 1)
        right_tree = self.construct_tree(X[best_split["right_mask"]], y[best_split["right_mask"]], depth + 1)

        return {
            "feature_index": best_split["feature_index"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree,
        }

    def predict_tree(self, x, tree):
        if "value" in tree:
            return tree["value"]

        if x[tree["feature_index"]] <= tree["threshold"]:
            return self.predict_tree(x, tree["left"])
        else:
            return self.predict_tree(x, tree["right"])

    def fit(self, X, y):
        self.initial_prediction = self.initialize_model_parameters(y)
        pred = np.full(y.shape, self.initial_prediction, dtype=np.float64)

        for _ in range(self.n_estimators):
            residuals = self.loss_gradient(y, pred)
            tree = self.fit_ensemble_tree(X, residuals)
            self.trees.append(tree)
            pred += self.learning_rate * np.array([self.predict_tree(x, tree) for x in X])

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_prediction, dtype=np.float64)

        for tree in self.trees:
            pred += self.learning_rate * np.array([self.predict_tree(x, tree) for x in X])

        return pred

    def r2_score_manual(self, y_true, y_pred):
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred)
        y_mean = np.mean(y_true)
        tss = np.sum((y_true - y_mean) ** 2)
        rss = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (rss / tss)
        return r2

    def mae_manual(self, y_true, y_pred):
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        return mae

    def rmse_manual(self, y_true, y_pred):
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred)
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse

def normalize_data(X):
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_normalized = (X - min_vals) / range_vals
    return X_normalized, min_vals, max_vals

def grid_search_gradient_boosting(X_train, y_train, X_test, y_test, n_estimators_values, learning_rate_values, max_depth_values):
    best_score = -float("inf")
    best_params = None

    for n_estimators in n_estimators_values:
        for learning_rate in learning_rate_values:
            for max_depth in max_depth_values:
                model = GradientBoostingTree(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = model.r2_score_manual(y_test, y_pred)
                if score > best_score:
                    best_score = score
                    best_params = {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                    }

    return best_params, best_score

def main():
    st.title("Gradient Boosting Regression App")
    st.write("""
    This app allows you to perform **Gradient Boosting Regression** on your dataset using a custom implementation.
    """)

    # Sidebar for hyperparameters
    st.sidebar.header("Hyperparameter Options")
    n_estimators_values = st.sidebar.multiselect(
        "Select n_estimators values",
        [10, 50, 100, 200, 300],
        default=[10, 50, 100]
    )
    learning_rate_values = st.sidebar.multiselect(
        "Select learning rate values",
        [0.1, 0.01, 0.001],
        default=[0.1, 0.01, 0.001]
    )
    max_depth_values = st.sidebar.multiselect(
        "Select max_depth values",
        [2, 3, 5, 7, 9],
        default=[2, 3, 5]
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON, or Parquet)", type=["csv", "xlsx", "json", "parquet"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                st.error("Unsupported file format. Please provide a CSV, Excel, JSON, or Parquet file.")
                return

            st.write("### Dataset Preview:")
            st.dataframe(df.head())

            target = st.selectbox("Select the target column", df.columns)

            if st.button("Run Model"):
                with st.spinner("Processing..."):

                    # Proceed with the code
                    check_null(df)
                    X, Y = XandY(df, target)

                    np.random.seed(42)
                    shuffled_indices = np.random.permutation(X.shape[0])
                    train_size = int(0.8 * len(shuffled_indices))
                    train_indices, test_indices = shuffled_indices[:train_size], shuffled_indices[train_size:]

                    X_train, X_test = X[train_indices], X[test_indices]
                    y_train, y_test = Y[train_indices], Y[test_indices]

                    st.write("Applying Normalization to the feature columns...")
                    X_train_normalized, train_min, train_max = normalize_data(X_train)
                    X_test_normalized, _, _ = normalize_data(X_test)
                    st.success("Pre-processing is Done.")

                    st.write("Using Grid Search to find the best parameters...")
                    st.write("Please wait, this might take some time depending on your CPU/GPU power...")

                    # Hyperparameter grids (from user selection)
                    if not n_estimators_values:
                        st.error("Please select at least one value for n_estimators.")
                        return
                    if not learning_rate_values:
                        st.error("Please select at least one value for learning rate.")
                        return
                    if not max_depth_values:
                        st.error("Please select at least one value for max_depth.")
                        return

                    best_params, best_score = grid_search_gradient_boosting(
                        X_train_normalized, y_train, X_test_normalized, y_test,
                        n_estimators_values, learning_rate_values, max_depth_values
                    )

                    st.write("### Best Parameters from Grid Search:")
                    st.write(f"- **n_estimators**: {best_params['n_estimators']}")
                    st.write(f"- **Learning Rate**: {best_params['learning_rate']}")
                    st.write(f"- **Max Depth**: {best_params['max_depth']}")
                    st.write(f"- **Best R² score**: {best_score:.4f}")

                    final_model = GradientBoostingTree(
                        n_estimators=best_params["n_estimators"],
                        learning_rate=best_params["learning_rate"],
                        max_depth=best_params["max_depth"]
                    )

                    final_model.fit(X_train_normalized, y_train)
                    y_pred_final = final_model.predict(X_test_normalized)

                    r2_final = final_model.r2_score_manual(y_test, y_pred_final)
                    mae_final = final_model.mae_manual(y_test, y_pred_final)
                    rmse_final = final_model.rmse_manual(y_test, y_pred_final)

                    st.write("### Final Model Evaluation:")
                    st.write(f"- **R² Score**: {r2_final:.4f}")
                    st.write(f"- **Mean Absolute Error (MAE)**: {mae_final:.4f}")
                    st.write(f"- **Root Mean Squared Error (RMSE)**: {rmse_final:.4f}")

                    y_test = np.array(y_test).ravel()
                    y_pred_final = np.array(y_pred_final).ravel()

                    # Plots
                    fig1, ax1 = plt.subplots()
                    sns.kdeplot(y_test, color='blue', fill=True, label='Actual Values', ax=ax1)
                    sns.kdeplot(y_pred_final, color='green', fill=True, label='Predicted Values', ax=ax1)
                    ax1.set_title('Density Plot of Actual vs Predicted Values')
                    ax1.set_xlabel('Values')
                    ax1.set_ylabel('Density')
                    ax1.legend()
                    ax1.grid(True)
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots()
                    ax2.scatter(y_test, y_pred_final, color='blue', label='Predicted Values', alpha=0.6)
                    ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
                             color='red', linestyle='--', label='Perfect Prediction')
                    ax2.set_xlabel('Actual Values')
                    ax2.set_ylabel('Predicted Values')
                    ax2.set_title('Prediction Error Plot')
                    ax2.legend()
                    ax2.grid(True)
                    st.pyplot(fig2)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

# Linear Regression Algorithm Implemented from scratch.

class LinearRegression:
    """
    Implemented from scratch using the normal equation.
    MSE calculation and Min-Max Scalng Utilities.
    """
    def __init__(self):
        # weights (coeffcientd) and bias (intercept) will be determined during fitting.
        self.weights = None
        self.bias = None
        self.min_vals = None #will store the minimum values for each feature for scaling.
        self.max_vals = None # will store the maximum values for each feature for scaling.
        self.min_target = None #will store the min value for target for scarling.
        self.max_target = None #stores max value for target for scaling.
        
    def _transpose(self, matrix):
        """
        Method for transposing matrices.
        """
        rows = len(matrix)
        cols = len(matrix[0])
        transposed = [[0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                transposed[j][i] = matrix[i][j]
        return transposed
    
    def _multiply_matrices(self, matrix1, matrix2):
        """
        Multiplies two matrices.
        """
        rows1 = len(matrix1)
        cols1 = len(matrix1[0])
        rows2 = len(matrix2)
        cols2 = len(matrix2[0])
        
        if cols1 != rows2:
            raise ValueError("Matrix dimensions are incopatible for multiplication.")
        
        result = [[0 for _ in range(cols2)] for _ in range(rows2)]
        for i in range(rows1):
            for j in range(cols2):
                for k in range(cols1):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        return result
    
    def _inverse_matrix(self, matrix):
        """
        Calculates the inverse of a (2x2) matrix.
        This will work for single feature linear regression.
        For higher dimensions we might need robust inverse calculations e.g (Gaussian elimination)
        """
        rows = len(matrix)
        cols = len(matrix[0])
        
        if rows != cols:
            raise ValueError("Cannot invert a non-square matrix")
        if rows == 1:
            return [[1 / matrix[0][0]]]
        if rows == 2:
            a, b = matrix[0][0], matrix[0][1]
            c, d = matrix[1][0], matrix[1][1]
            determinant = a * d - b * c
            if determinant == 0:
                raise ValueError("Matrix is singular and cannot be inverted.")
            inv_det = 1 / determinant
            return [[d * inv_det, -b * inv_det],
                    [-c * inv_det, a * inv_det]]
        else: 
            raise NotImplementedError("Matrix inversion for dimensions > 2x2 not implemented.")
        
    def _add_bias_term(self, X):
        """
        will add a column of ones to the feature matrix X to account for the bias (X intercept).
        Necessary for the normal equation where the bias term is treated as a weight,
        associated with a constant feature of 1.
        """
        num_samples = len(X)
        # creating the columns of ones.
        ones_column = [[1] for _ in range(num_samples)]
        
        # now we concatenate ones_column with X, assuming X is a matrix (list of lists)
        X_b = []
        for i in range(num_samples):
            X_b.append(ones_column[i] + X[i])
        return X_b
    
    def _mean_squared_error(self, y_true, y_pred):
        """
        calculates mse between true and predicted values.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("True and predicted arrays must have the same length.")
        n = len(y_true)
        if n == 0:
            return 0.0
        
        squared_errors_sum = 0
        for i in range(n):
            squared_errors_sum += (y_true[i] - y_pred[i]) ** 2
        return squared_errors_sum / n
    
    def _scale_data(self, data, fit_scaler=True, is_target=False):
        """
        will apply the min-max scaling to the input data.
        if fit_scaler is True, it calculates and stores min/max values.
        If is_target is True, it scales the 1D target Array.
        """
        if not data:
            return []
        if is_target:
            # handling the 1D target array.
            if fit_scaler:
                self.min_target = data[0]
                self.max_target = data[0]
                for val in data:
                    if val < self.min_target:
                        self.min_target = val
                    if val > self.max_target:
                        self.max_target = val
            if self.min_target is None or self.max_target is None:
                raise ValueError("Scaler not fitted for target data. Call with fit_scaler")
            scaled_data = []
            range_target = self.max_target - self.min_target
            if range_target == 0: #avoid division by 0.
                return [0.0] * len(data)
            for val in data:
                scaled_data.append((val - self.min_target) / range_target)
            return scaled_data
        else:
            # Handle 2D feature matrix
            num_features = len(data[0])

            if fit_scaler:
                self.min_vals = [float('inf')] * num_features
                self.max_vals = [float('-inf')] * num_features
                for row in data:
                    for j in range(num_features):
                        if row[j] < self.min_vals[j]:
                            self.min_vals[j] = row[j]
                        if row[j] > self.max_vals[j]:
                            self.max_vals[j] = row[j]

            if self.min_vals is None or self.max_vals is None:
                raise ValueError("Scaler not fitted for feature data. Call with fit_scaler=True first.")

            scaled_data = []
            for row in data:
                scaled_row = []
                for j in range(num_features):
                    feature_range = self.max_vals[j] - self.min_vals[j]
                    if feature_range == 0: # Avoid division by zero if all values for a feature are the same
                        scaled_row.append(0.0)
                    else:
                        scaled_row.append((row[j] - self.min_vals[j]) / feature_range)
                scaled_data.append(scaled_row)
            return scaled_data
        
    def _inverse_scale_data(self, scaled_data, is_target=False):
        """
        Reverses Min-Max scaling to get original data values.
        If is_target is True, it inverse scales a 1D target array.
        """
        if not scaled_data:
            return []

        if is_target:
            if self.min_target is None or self.max_target is None:
                raise ValueError("Scaler not fitted for target data. Cannot inverse scale.")

            original_data = []
            range_target = self.max_target - self.min_target
            for val in scaled_data:
                original_data.append(val * range_target + self.min_target)
            return original_data
        else:
            if self.min_vals is None or self.max_vals is None:
                raise ValueError("Scaler not fitted for feature data. Cannot inverse scale.")

            original_data = []
            num_features = len(scaled_data[0])
            for row in scaled_data:
                original_row = []
                for j in range(num_features):
                    feature_range = self.max_vals[j] - self.min_vals[j]
                    original_row.append(row[j] * feature_range + self.min_vals[j])
                original_data.append(original_row)
            return original_data

    def fit(self, X, y):
        """
        Trains the Linear Regression model using the Normal Equation.
        The normal equation formula is: theta = (X.T * X)^-1 * X.T * y
        where X is the design matrix (features + bias term) and y is the target vector.

        Args:
            X (list of lists): The feature matrix (e.g., [[feature1_1, feature1_2], ...]).
            y (list): The target vector (e.g., [target1, target2, ...]).
        """
        if not X or not y:
            raise ValueError("Input features X and target y cannot be empty.")
        if len(X) != len(y):
            raise ValueError("Number of samples in X and y must be the same.")

        # Scale features and target
        X_scaled = self._scale_data(X, fit_scaler=True, is_target=False)
        y_scaled = self._scale_data(y, fit_scaler=True, is_target=True)

        # Add bias term (column of ones) to X_scaled
        X_b = self._add_bias_term(X_scaled)

        # Convert y_scaled to a column vector (list of lists) for matrix multiplication
        y_col_vector = [[val] for val in y_scaled]

        # Calculate (X.T * X)
        X_b_T = self._transpose(X_b)
        Xt_X = self._multiply_matrices(X_b_T, X_b)

        # Calculate (X.T * X)^-1
        try:
            Xt_X_inv = self._inverse_matrix(Xt_X)
        except NotImplementedError as e:
            print(f"Warning: {e}. Normal equation might not work for multi-feature data without a full matrix inverse implementation.")
            print("Attempting to proceed, but results may be inaccurate if X.T @ X is not 2x2.")
            # For simplicity and to allow single-feature cases to pass,
            # we'll just re-raise if it's truly an unhandled dimension.
            raise

        # Calculate (X.T * y)
        Xt_y = self._multiply_matrices(X_b_T, y_col_vector)

        # Calculate theta = (X.T * X)^-1 * X.T * y
        theta = self._multiply_matrices(Xt_X_inv, Xt_y)

        # The first element of theta is the bias, the rest are weights
        self.bias = theta[0][0]
        self.weights = [theta[i][0] for i in range(1, len(theta))]

    def predict(self, X):
        """
        Makes predictions using the learned weights and bias.

        Args:
            X (list of lists): The feature matrix for which to make predictions.

        Returns:
            list: The predicted target values.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if not X:
            return []

        # Scale the input features using the previously fitted scaler
        X_scaled = self._scale_data(X, fit_scaler=False, is_target=False)

        predictions_scaled = []
        for i in range(len(X_scaled)):
            prediction_i = self.bias
            for j in range(len(self.weights)):
                prediction_i += self.weights[j] * X_scaled[i][j]
            predictions_scaled.append(prediction_i)

        # Inverse scale the predictions to get values in the original target range
        predictions_original_scale = self._inverse_scale_data(predictions_scaled, is_target=True)
        return predictions_original_scale

    def evaluate(self, y_true, y_pred):
        """
        Evaluates the model by calculating the Mean Squared Error (MSE).

        Args:
            y_true (list): The true target values.
            y_pred (list): The predicted target values.

        Returns:
            float: The Mean Squared Error.
        """
        return self._mean_squared_error(y_true, y_pred)

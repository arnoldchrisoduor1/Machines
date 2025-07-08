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
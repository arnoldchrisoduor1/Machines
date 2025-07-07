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
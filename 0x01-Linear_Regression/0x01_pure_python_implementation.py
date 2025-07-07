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
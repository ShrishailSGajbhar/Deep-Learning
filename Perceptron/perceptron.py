from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Tuple, List

class Perceptron:
    def __init__(self, X: List[List[float]], y: List[int], b: float) -> None:
        """Perceptron class to understand the learning algorithm

        Args:
            X (list): input features e.g., [x1,x2,x3]
            y (list): target variable
            b (float): threshold 
        """
        self.X, self.y = self._check_data(X, y)
        self.b = b 
        # Initialize the weight vector
        self.w = np.ones(self.X.shape[0])

    def _check_data(self, x, y):
        """Load the data for perceptron

        Args:
            x (np.array): vector array representing the input features
            y (np.array): vector array representing the target feature
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        return x, y
     
    @staticmethod
    def abline(slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')

        
    def plot_data(self, w: np.ndarray, b: float, xlim: Optional[Tuple[float, float]] = None, ylim: Optional[Tuple[float, float]] = None, title: str = "Perceptron input data (x1 and x2 features only)"):
        """Plot the points to be classified by perceptron algorithm using matplotlib
        Args:
            w (np.ndarray): weight vector
            b (float): bias term
            xlim (Optional[Tuple[float, float]]): x axis limits
            ylim (Optional[Tuple[float, float]]): y axis limits  
        """         
        colormap = ["red", "green"]
        x1 = self.X[0]
        x2 = self.X[1]
        plt.scatter(x1, x2, c=[colormap[label] for label in self.y])
        plt.xlabel("x1")
        plt.xlim(xlim)
        plt.ylabel("x2")
        plt.ylim(ylim)
        plt.title(title)
        # calculate the slope and intercept
        m = -w[0] / w[1]
        c = b / w[1]
        Perceptron.abline(m, c)
        plt.draw()
        # plt.pause(0.5)
        # plt.clf()
        plt.show()

    def indicator_loss(self, y, yhat):
        """Calculate the indicator loss

        Args:
            y (np.array): True labels
            yhat (np.array): Predicted labels

        Returns:
            float: Indicator loss (0 if all predictions are correct, 1 if any prediction is incorrect)
        """
        # Ensure inputs are numpy arrays
        y = np.array(y)
        yhat = np.array(yhat)
        
        # Calculate the indicator loss
        loss = np.mean(y != yhat)
    
        return loss

    def learning_algorithm(self):
        """Perceptron learning algorithm
        """
        # Add a bias term to the input data
        Xmat = np.vstack((self.X, np.ones(self.X.shape[1])))
        Wmat = np.append(self.w, self.b)
        yhat = np.dot(Wmat, Xmat) > self.b
        yhat_int = yhat.astype(int)  # Convert boolean array to integers (1 for True, 0 for False)
        loss = self.indicator_loss(self.y, yhat_int)
        
        iteration = 0
        while loss > 1e-3:
            # Loop till loss function is less than some epsilon
            self.plot_data(w=Wmat[:-1], b=Wmat[-1], xlim=(-20, 20), ylim=(-20, 20), title=f"Iteration {iteration}")
            iteration += 1
            error_indices = np.where(self.y != yhat_int)[0]
            for idx in error_indices:
                if self.y[idx] == 1:  # Positive point
                    Wmat[:-1] += Xmat[:-1, idx]
                    Wmat[-1] += 1  # Update bias
                else:  # Negative point
                    Wmat[:-1] -= Xmat[:-1, idx]
                    Wmat[-1] -= 1  # Update bias
                yhat = np.dot(Wmat, Xmat) > self.b
                yhat_int = yhat.astype(int)
                loss = self.indicator_loss(self.y, yhat_int)
                
        self.w = Wmat[:-1]
        self.b = Wmat[-1]
        print("Algorithm converged")
        self.plot_data(w = self.w, b = self.b, xlim=(-20,20), ylim=(-20,20), title="After convergence seperator line")



if __name__ == "__main__":
    X = [[-1, -5, -7.5, 10, -2.5, 5, 5], [-1, -2.5, 7.5, 7.5, 12.5, 10, 5]]
    y = [0, 0, 0, 1, 0, 1, 1]
    
    p = Perceptron(X, y, 5.0)
    p.learning_algorithm()
    print("Indicator loss for y=yhat", p.indicator_loss(y, y))

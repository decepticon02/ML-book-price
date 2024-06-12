from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle


class My_Logistic_Regression:
    def __init__(self, alpha, max_iter, eta0, epsilon=0.0001):
        self.alpha = alpha  
        self.max_iter = max_iter 
        self.eta0 = eta0  
        self.epsilon = epsilon  
        self.loss = []  
        self.accuracy = []  
        self.weights = []  
        self.intercept_ = []  

    def sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))  

    def fit(self, x, y):
        malo=0.001
        num_samples, num_features = x.shape
        

        for j in np.unique(y):
            weights_temp = np.random.normal(loc=0, scale=0.0001, size=num_features)
            intercept = 0
            class_loss=[]
            class_accuracy=[]
            for _ in range(self.max_iter):
                x_shuffled, y_shuffled = shuffle(x, (y == j).astype(int), random_state=42)

                loss_tmp = 0
                correct_predictions = 0

                for i in range(num_samples):
                    x_i = x_shuffled[i]
                    y_i = y_shuffled[i]

                    z = np.dot(x_i, weights_temp) + intercept
                    z_clipped = np.clip(z, -1000, 1000)  
                    y_pred = self.sigmoid(z_clipped)

                    loss_tmp += -(y_i * np.log(y_pred ) + (1 - y_i) * np.log(1 - y_pred + malo))

                    error = y_pred - y_i
                    weights_temp -= self.eta0 * (error * x_i + self.alpha * weights_temp)
                    intercept -= self.eta0 * error

                    if (y_pred >= 0.5 and y_i == 1) or (y_pred < 0.5 and y_i == 0):
                        correct_predictions += 1

                loss_tmp /= num_samples
                accuracy_temp = correct_predictions / num_samples
                class_loss.append(loss_tmp)
                class_accuracy.append(accuracy_temp)
            
                print(f"Class {j} - Iteration {_ + 1}/{self.max_iter} - Accuracy: {accuracy_temp}")

                if _ > 0 and abs(loss_tmp - class_loss[-2]) < self.epsilon:
                    print(f"Converged for class {j} at iteration {_ + 1}")
                    break
            self.loss.append(class_loss)
            self.accuracy.append(class_accuracy)
            self.weights.append(weights_temp)
            self.intercept_.append(intercept)

     
                      

    def predict(self, x):
        num_samples = x.shape[0]
        num_classes = len(self.weights)
        class_probabilities = np.zeros((num_samples, num_classes))
       
        
        for j, (weights, intercept) in enumerate(zip(self.weights, self.intercept_)):
            z = np.dot(x, weights) + intercept
            z_clipped = np.clip(z, -1000, 1000)  
            class_probabilities[:, j] = self.sigmoid(z_clipped)

        predicted_classes = np.argmax(class_probabilities, axis=1)+1
       
        return predicted_classes

    # def plot_loss(self):
        
    #     plt.plot(range(0,6),self.loss)
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Loss')
    #     plt.title('Loss over iterations')
    #     plt.show()

    #     plt.plot(range(0,6),self.accuracy)
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Accuracy')
    #     plt.title('Accuracy over iterations')
    #     plt.show()

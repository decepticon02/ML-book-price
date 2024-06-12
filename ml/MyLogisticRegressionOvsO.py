from itertools import combinations
import numpy as np
from sklearn.utils import shuffle


class My_OneVsOne_Logistic_Regression:
    def __init__(self, alpha, max_iter, eta0, epsilon=0.0001):
        self.alpha = alpha  
        self.max_iter = max_iter 
        self.eta0 = eta0  
        self.epsilon = epsilon  
        self.loss = []  
        self.accuracy = []  
        self.weights = []  
        self.intercept_ = {}  
        self.class_pairs = []  
        self.classifiers = {} 
        self.positiveclasses={}

    def sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))  

    def fit(self, x, y):
        malo=0.001
        
        classes = np.unique(y)
        

        self.class_pairs = list(combinations(classes, 2))

        for pair in self.class_pairs:
            class_1, class_2 = pair
            x_pair = x[(y == class_1) | (y == class_2)]
            y_pair = y[(y == class_1) | (y == class_2)]
            y_lbl=class_1
            if(len(y[(y == class_1)])>len(y[(y == class_2)])):
                y_lbl=class_2
            self.positiveclasses[pair]=y_lbl
            y_pair = np.where((y_pair == y_lbl), 1, 0)  
            num_samples, num_features = x_pair.shape
            weights_temp = np.random.normal(loc=0, scale=0.0001, size=num_features)
            intercept=0
            class_loss=[]
            class_accuracy=[]

            for _ in range(self.max_iter):
                x_shuffled, y_shuffled = shuffle(x_pair, y_pair, random_state=42)

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
            
                print(f"Classiffier {pair} - Iteration {_ + 1}/{self.max_iter} - Accuracy: {accuracy_temp}")

                if _ > 0 and abs(loss_tmp - class_loss[-2]) < self.epsilon:
                    print(f"Converged for classifier {pair} at iteration {_ + 1}")
                    break
            self.intercept_[pair]=intercept
            self.classifiers[pair] = weights_temp

    def predict(self, x):
        num_samples = x.shape[0]
        predictions = np.zeros((num_samples, len(self.class_pairs)))

        for i, pair in enumerate(self.class_pairs):
            z = np.dot(x, self.classifiers[pair]) + self.intercept_[pair]
            z_clipped = np.clip(z, -1000, 1000)
            predictions[:, i] = np.where(self.sigmoid(z_clipped) >= 0.5, self.positiveclasses[pair],np.where((self.positiveclasses[pair]==pair[0]),pair[1],pair[0]))

        final_predictions = []
        for preds in predictions:
            final_predictions.append(np.argmax(np.bincount(preds.astype(int))))

        return np.array(final_predictions)

import matplotlib.pyplot as plt
from numpy import asarray
import numpy as np
import random
from sklearn.utils import shuffle


class My_Linear_Regression:
# implementirana tako da koristi l2 regularizaciju!
    def __init__(self, alpha, epsilon, eta0,  tol, max_iter):
        self.alpha = alpha # koef uz l2
        self.epsilon = epsilon #stajemo sa ucenjem kada greska postane manja od episilon 
        self.eta0 = eta0 #pocetna brzina ucenja 
        self.tol = tol # svako smanjivanje greske po ciklusim amora da bude vece od tol
        self.max_iter = max_iter 
        self.learning_rate = 'adaptive' #samo reda radi stavljeno
        self.n_iter_no_change=5 # broj iteracija u kojima se dozvoljava da stagnira greska
        self.loss=[] #kako bih crtala posle fju greske po iteracjiama
        self.num_epochs=[] #zbog iscrtavanja glupo 
        self.accuracy_epoch=[] #zbog iscrtavanja glupo 
        

    def fit(self, x, y):
        rows, features = x.shape
        self.weights = np.random.normal(loc=0, scale=0.01, size=(1, features)) # male ne nulte vrednosti
        self.intercept_ = 0 # w0
        
        eta = self.eta0 #konstatno dok ne krene da "previse" stagrnira smanjivanje greske
        
        #y = y.to_numpy() 
        consecutive_no_change = 0  
        best_loss = float('inf') 

        for _ in range(self.max_iter):
            
            x_shuffled, y_shuffled = shuffle(x,y,random_state=0) 
            cost = 0
            correct_predictions=0
            for i in range(rows):

                x_i = x_shuffled[i]
                y_i = y_shuffled[i]

                y_pred = np.dot(x_i, self.weights.T) + self.intercept_ 
                error = y_pred - y_i
                loss = (((error[0]) ** 2) )/ (2*rows)
                cost += loss

                self.weights -= eta * (error[0] * x_i) / rows + eta*self.alpha * self.weights / rows
             
                if abs(error) < self.tol:  # Ako je apsolutna razlika manja od tolerancije, smatra se tačnom predikcijom
                    correct_predictions += 1

                self.intercept_ -= eta * error[0]/rows

            cost += (self.alpha * np.sum(self.weights ** 2)) / (2 * rows) # L2 reg

            if cost < self.epsilon: #sadasnja greska manja od minimalnog praga za gresku gotovi sa ucenjem
                self.loss.append(cost)  
                break
            # if cost < best_loss - self.tol:  # smanjili smo gresku dodatno
            #     best_loss = cost
            #     consecutive_no_change = 0
            # else:  # nismo napravili smanjenje greske u ovoj iteraciji 
            #     consecutive_no_change += 1
            #     if consecutive_no_change == self.n_iter_no_change: 
            #         eta /=5  #smanjujemo brzinu ucenja
            #         print("smanjila")
            #         consecutive_no_change = 0

            self.loss.append(cost)  
            self.num_epochs.append(_) #samo zbog iscrtavanja :(
            acc = correct_predictions / rows
            self.accuracy_epoch.append(acc)
            print(f"Epoch {_+1}/{self.max_iter} - Loss: {cost}, Accuracy: {acc}")

        self.weights=self.weights.reshape(-1) 


    def predict(self, x):
        return np.dot(x, self.weights) + self.intercept_
    

    def plotLoss(self):
        plt.plot(self.num_epochs, self.loss)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.title('Plot Loss')
        plt.show()
        plt.plot(self.accuracy_epoch, self.loss)
        plt.xlabel('Number of Epochs')
        plt.ylabel('accuracy')
        plt.title('Plot acc')
        plt.show()



  
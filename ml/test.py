from sklearn.utils import shuffle
import numpy as np
# Sample data
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([10, 20, 30])

# Shuffle the data
x_shuffled, y_shuffled = shuffle(x, y, random_state=0)

for i in range(x.shape[0]):
    print(x[i],y[i])


# Print the shuffled arrays
print("Shuffled x:")
print(x_shuffled)
print("\nShuffled y:")
print(y_shuffled)

##################
# load iris
##################

from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

##################
# onehot encoding
##################

from tensorflow.keras.utils import to_categorical

y_onehot = to_categorical(y)
print(y_onehot)

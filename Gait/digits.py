from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
print digits.data[1],digits.target[1]

plt.gray()
plt.matshow(digits.images[-1])
plt.show()
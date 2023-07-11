"""
Datasets.

@author: Arun Kr. Khattri
         arun.kr.khattri@gmail.com
"""
from sklearn.datasets import load_digits, load_breast_cancer, load_diabetes

import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA

# matplotlib.use("qtagg")


# ===========================================================
# DATASETS
# ===========================================================
diabetes = load_diabetes()
bc = load_breast_cancer()
digits = load_digits()

images_and_labels = list(zip(digits.images, digits.target))

for idx, (img, label) in enumerate(images_and_labels[10:20]):
    plt.subplot(2, 5, idx + 1)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f"Target: {label}")

plt.show()

# ===========================================================
# CLASSIFICATION
# ===========================================================
f = lambda x: 2 * x - 5

pos = []
neg = []

for i in range(30):
    x = np.random.randint(15)
    y = np.random.randint(15)

    if f(x) < y:
        pos.append([x, y])
    else:
        neg.append([x, y])

plt.figure()
plt.xticks([])
plt.yticks([])
plt.scatter(*zip(*pos))
plt.scatter(*zip(*neg))
plt.plot([0, 10], [f(0), f(10)], linestyle='--', color='m')
plt.xlabel('x')
plt.ylabel('y')
plt.title('classification')
plt.show()

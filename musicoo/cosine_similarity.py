from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vector1 = np.array([1, 0, 0])
vector2 = np.array([0, 1, 0])

vector1 = vector1.reshape(1, -1)
vector2 = vector2.reshape(1, -1)

similarity = cosine_similarity(vector1, vector2)

print(similarity)
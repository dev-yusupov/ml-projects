# Confusion matrices for each model
confusion_matrices = {
    "KNN": [[106, 29], [36, 109]],
    "Random Forest": [[104, 31], [24, 121]],
    "Decision Tree": [[97, 38], [27, 118]],
    "Gradient Boosting": [[113, 22], [25, 120]],
    "Ensemble": [[107, 28], [30, 115]],
    "Logistic": [[90, 45], [42, 103]],
    "SVC": [[105, 30], [46, 99]],
    "Ada": [[102, 33], [28, 117]]
}

# Calculate precision and accuracy for each model
for model, matrix in confusion_matrices.items():
    TP = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TN = matrix[1][1]

    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"Precision for {model}: {precision}")
    print(f"Accuracy for {model}: {accuracy}")

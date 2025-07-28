# Task : Predicting Answer Correctness Using Neural Networks

## Objective

The goal of this task is to build a machine learning model—specifically a neural network—that predicts whether a user's answer to a multiple-choice question is correct. You will use the provided datasets (`train.pickle` and `test.pickle`) to train and evaluate your model.

---

## Files Provided

- `train.pickle`: Contains training data with features representing user attempts at answering questions.
- `test.pickle`: Contains test data with the same structure as the training data.
- `README.md`: You will overwrite this file with your experimental report.

---

## Task Breakdown

### Part 1: Model Development

1. **Goal**: Train a neural network to predict the `is_correct` column in the test set.
2. **Reproducibility**: Your code must produce the same results every time it is run on the same machine. This means:
   - Set all random seeds (NumPy, PyTorch/TensorFlow, etc.)
   - Avoid non-deterministic operations unless controlled
3. **Performance Target**: Aim for **95%+ accuracy** on the test set. This is achievable.
4. **Hint**: One of the features encodes the answer the user selected—this is crucial for prediction.

### Part 2: Experimental Report

You must overwrite `README.md` with a detailed report that includes:

- **Approach**: Describe your model architecture, preprocessing steps, and training strategy.
- **Experiments**: What variations did you try? (e.g., different architectures, optimizers, feature engineering)
- **Results**: Report accuracy and other relevant metrics.
- **Insights**: What worked well? What didn’t? What would you try next?
- **Reproducibility**: Mention how you ensured consistent results.

---

## Evaluation Criteria

| Criterion              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Accuracy               | How well your model predicts `is_correct` on the test set                   |
| Reproducibility        | Can your results be reproduced exactly?                                     |
| Code Quality           | Is your code clean, modular, and well-documented?                           |
| Report Quality         | Is your README.md clear, insightful, and comprehensive?                     |
| Creativity & Rigor     | Did you explore different ideas and justify your choices?                   |

---

## Tips

- Use standard libraries like PyTorch, TensorFlow, or scikit-learn.
- Consider normalizing or encoding features if needed.
- Use validation splits or cross-validation to tune hyperparameters.
- Log your experiments for easier comparison.


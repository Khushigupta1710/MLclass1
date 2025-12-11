import numpy as np

# -----------------------------
# Data (replace with your values)
# -----------------------------
x1 = np.array([5, 3, 4, 3, 5])
x2 = np.array([3, 2, 4, 1, 3])
y  = np.array([35, 25, 30, 20, 32])

# -----------------------------
# Create design matrix X
# X = [1, x1, x2]
# -----------------------------
n = len(x1)
X = np.column_stack((np.ones(n), x1, x2))

# -----------------------------
# Apply Normal Equation:
# β = (Xᵀ X)⁻¹ Xᵀ y
# -----------------------------
XtX = X.T.dot(X)
XtY = X.T.dot(y)

beta = np.linalg.inv(XtX).dot(XtY)

b0, b1, b2 = beta

print("β0 (intercept):", b0)
print("β1 (coefficient for x1):", b1)
print("β2 (coefficient for x2):", b2)

# -----------------------------
# Prediction (your notebook used x1=3, x2=2)
# -----------------------------
x1_new = 3
x2_new = 2

y_pred = b0 + b1*x1_new + b2*x2_new
print("\nPredicted y for x1=3, x2=2:", y_pred)
 pip install numpy
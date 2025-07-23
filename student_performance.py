import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv("student-mat.csv")  # or your local dataset path
print(df.head())
print(df.describe())
sns.pairplot(df[['G1', 'G2', 'G3', 'studytime', 'absences']])
# Convert categorical columns to numerical
df = pd.get_dummies(df, drop_first=True)

# Select features and target
X = df.drop("G3", axis=1)  # G3 = final grade
y = df["G3"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
plt.scatter(y_test, predictions)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")
plt.show()

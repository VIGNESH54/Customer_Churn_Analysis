import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Dataset
dataset_path = os.path.expanduser("~/Desktop/Customer_Churn_Analysis/data/customer_churn.csv")
data = pd.read_csv(dataset_path)

# Display first few rows
print("✅ Dataset Loaded Successfully!")
print(data.head())

# Ensure images folder exists
images_folder = os.path.expanduser("~/Desktop/Customer_Churn_Analysis/images")
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

# ---------------------------
# 🔹 Churn Distribution
# ---------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=data, palette="coolwarm")
plt.title("Churn Distribution (0 = Stayed, 1 = Left)")
plt.savefig(os.path.join(images_folder, "churn_distribution.png"))
plt.show()

# ---------------------------
# 🔹 Feature Importance Analysis
# ---------------------------
# Calculate average values for Churn vs. Non-Churn customers
churned = data[data["Churn"] == 1].mean()
not_churned = data[data["Churn"] == 0].mean()
features = ["Age", "Subscription_Length_Months", "Monthly_Charges", "Total_Charges", "Support_Calls"]

# Plot differences
plt.figure(figsize=(8, 5))
for feature in features:
    plt.bar(feature, churned[feature] - not_churned[feature], color="red" if churned[feature] > not_churned[feature] else "green")

plt.axhline(0, color="black", linestyle="dashed")
plt.title("Feature Importance in Churn")
plt.ylabel("Difference Between Churned & Non-Churned")
plt.savefig(os.path.join(images_folder, "feature_importance.png"))
plt.show()

print("\n✅ Analysis Completed! All charts saved in images/ folder.")

from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
import os

app = Flask(__name__)

# --------------------------------
# 1. Load customer dataset
# --------------------------------
df = pd.read_csv("customer_data.csv")

# Check required columns
required_columns = [
    "CustomerID",
    "Annual_Income",
    "Spending_Score"
]

if not all(column in df.columns for column in required_columns):
    raise ValueError(
        "CSV must contain CustomerID, Annual_Income, "
        "and Spending_Score columns."
    )

# --------------------------------
# 2. Prepare data
# --------------------------------
X = df[["Annual_Income", "Spending_Score"]]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------
# 3. Train K-Means model
# --------------------------------
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# IMPORTANT: Create Cluster column before plotting
df["Cluster"] = model.fit_predict(X_scaled)

# --------------------------------
# 4. Create clustering graph
# --------------------------------
fig = Figure(figsize=(8, 5))
ax = fig.subplots()

# Plot customers in different colors
for cluster_id in sorted(df["Cluster"].unique()):
    cluster_data = df[df["Cluster"] == cluster_id]

    ax.scatter(
        cluster_data["Annual_Income"],
        cluster_data["Spending_Score"],
        label=f"Cluster {cluster_id}",
        s=90,
        alpha=0.8
    )

# Plot cluster centers in original feature units
centers = scaler.inverse_transform(model.cluster_centers_)

ax.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200,
    color="black",
    label="Cluster Centers"
)

ax.set_title("Customer Segmentation using K-Means")
ax.set_xlabel("Annual Income (thousands)")
ax.set_ylabel("Spending Score")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)

# Ensure static folder exists
static_folder = os.path.join(app.root_path, "static")
os.makedirs(static_folder, exist_ok=True)

# Save the graph
plot_path = os.path.join(static_folder, "cluster_plot.png")

fig.savefig(
    plot_path,
    format="png",
    dpi=120,
    bbox_inches="tight"
)

# --------------------------------
# 5. Flask route
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None
    income = ""
    spending = ""

    if request.method == "POST":

        income = request.form.get("income", "").strip()
        spending = request.form.get("spending", "").strip()

        # Required-field validation
        if not income or not spending:
            error = "All fields are required."

        else:
            try:
                income_value = float(income)
                spending_value = float(spending)

                if income_value <= 0:
                    error = "Annual income must be greater than zero."

                elif not 1 <= spending_value <= 100:
                    error = "Spending score must be between 1 and 100."

                else:
                    # Prepare new customer data
                    new_customer = pd.DataFrame(
                        [[income_value, spending_value]],
                        columns=[
                            "Annual_Income",
                            "Spending_Score"
                        ]
                    )

                    # Apply the same scaler used during training
                    new_scaled = scaler.transform(new_customer)

                    # Predict cluster
                    result = int(model.predict(new_scaled)[0])

            except ValueError:
                error = "Please enter valid numeric values."

    # Display webpage
    return render_template(
        "index.html",
        result=result,
        error=error,
        income=income,
        spending=spending,
        customers=df.to_dict(orient="records")
    )


# --------------------------------
# 6. Run Flask application
# --------------------------------
if __name__ == "__main__":
    app.run(debug=True)
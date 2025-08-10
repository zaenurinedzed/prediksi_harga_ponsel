import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("phone_price_dataset.csv")

# Pisahkan fitur dan target
X = df.drop("price_range", axis=1)
y = df["price_range"]

# Bagi data latih dan data uji (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Latih model
model.fit(X_train, y_train)

# Prediksi data uji
y_pred = model.predict(X_test)

# Evaluasi hasil prediksi
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Contoh prediksi dengan data baru
data_baru = {
    "battery_power": [2600],
    "blue": [1],
    "clock_speed": [1.9],
    "dual_sim": [1],
    "fc": [5],
    "pc": [16],
    "ram": [2048]
}

df_baru = pd.DataFrame(data_baru)
prediksi = model.predict(df_baru)
print("\nPrediksi kelas harga untuk data baru:", prediksi[0])

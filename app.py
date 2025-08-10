from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_data = None
    if request.method == "POST":
        battery_power = int(request.form.get("battery_power"))
        blue = int(request.form.get("blue"))
        clock_speed = float(request.form.get("clock_speed"))
        dual_sim = int(request.form.get("dual_sim"))
        fc = int(request.form.get("fc"))
        pc = int(request.form.get("pc"))
        ram = int(request.form.get("ram"))

        input_data = {
            "battery_power": battery_power,
            "blue": blue,
            "clock_speed": clock_speed,
            "dual_sim": dual_sim,
            "fc": fc,
            "pc": pc,
            "ram": ram
        }

        data = pd.DataFrame({
            "battery_power": [battery_power],
            "blue": [blue],
            "clock_speed": [clock_speed],
            "dual_sim": [dual_sim],
            "fc": [fc],
            "pc": [pc],
            "ram": [ram]
        })

        pred = model.predict(data)[0]
        price_map = {0: "Murah", 1: "Menengah Bawah", 2: "Menengah Atas", 3: "Mahal"}
        prediction = price_map.get(pred, "Tidak diketahui")

    return render_template("index.html", prediction=prediction, input_data=input_data)

if __name__ == "__main__":
    app.run(debug=True)

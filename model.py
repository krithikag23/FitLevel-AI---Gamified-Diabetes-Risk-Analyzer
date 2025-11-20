import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load built-in diabetes dataset
_diabetes = load_diabetes()
X = _diabetes.data
y = _diabetes.target
FEATURE_NAMES = _diabetes.feature_names  # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

_model = RandomForestRegressor(n_estimators=200, random_state=42)
_model.fit(X_train, y_train)

# Normalize target range
Y_MIN = float(y_train.min())
Y_MAX = float(y_train.max())

# Feature stats for slider mapping
_feature_mins = X.min(axis=0)
_feature_maxs = X.max(axis=0)
_feature_means = X.mean(axis=0)

FEATURE_STATS = {
    name: {
        "min": _feature_mins[i],
        "max": _feature_maxs[i],
        "mean": _feature_means[i],
    }
    for i, name in enumerate(FEATURE_NAMES)
}

# Controllable features
CONTROLLED_FEATURES = {
    "age": "Age index",
    "bmi": "Body fitness (BMI index)",
    "bp": "Blood pressure index",
    "s1": "Cholesterol index",
    "s5": "Triglycerides index",
    "s6": "Blood sugar index",
}

def _index_to_value(feature_name: str, index_0_100: float) -> float:
    stats = FEATURE_STATS[feature_name]
    f_min, f_max = stats["min"], stats["max"]
    return f_min + (f_max - f_min) * (index_0_100 / 100.0)

def build_input_vector(slider_values: dict) -> np.ndarray:
    vals = []
    for name in FEATURE_NAMES:
        if name in CONTROLLED_FEATURES:
            idx = slider_values.get(name, 50)
            vals.append(_index_to_value(name, idx))
        else:
            vals.append(FEATURE_STATS[name]["mean"])
    return np.array(vals, dtype=float).reshape(1, -1)

def predict_risk(slider_values: dict):
    x = build_input_vector(slider_values)
    pred = float(_model.predict(x)[0])
    risk = (pred - Y_MIN) / (Y_MAX - Y_MIN + 1e-8)
    return pred, round(max(0.0, min(1.0, risk)) * 100, 1)

def get_feature_importance():
    imp = _model.feature_importances_
    return sorted(zip(FEATURE_NAMES, imp), key=lambda x: x[1], reverse=True)

def get_gamified_level(score):
    if score < 30:
        return {"title": "Health Rookie 🌱", "color": "green", "message": "Great start! Maintain consistency!"}
    elif score < 60:
        return {"title": "Balance Seeker ⚖️", "color": "yellow", "message": "Some tweaks will level you up fast!"}
    elif score < 80:
        return {"title": "Risk Ranger 🔥", "color": "orange", "message": "Time to improve lifestyle habits!"}
    else:
        return {"title": "Boss Level Alert 🚨", "color": "red", "message": "High risk! Please consider medical guidance."}

def get_lifestyle_quests(slider_values):
    quests = []
    age, bmi, bp = slider_values.get("age",50), slider_values.get("bmi",50), slider_values.get("bp",50)
    chol, trig, glu = slider_values.get("s1",50), slider_values.get("s5",50), slider_values.get("s6",50)

    if bmi > 60:
        quests += ["Walk 30 mins/day 🏃‍♀️", "Swap sugary snacks with fruits 🍎"]
    if bp > 60:
        quests += ["Reduce salt 🍲", "10 mins meditation/day 🧘‍♀️"]
    if chol > 60:
        quests += ["Avoid fried food 🍟", "Eat fiber rich meals 🥬"]
    if trig > 60:
        quests += ["Replace soda with water 🚰", "Avoid late-night meals 🌙"]
    if glu > 60:
        quests += ["Consistent meal timing ⏱️", "Short walks after meals 🚶‍♀️"]
    if age > 60:
        quests += ["Regular checkups 👩‍⚕️", "Strength exercises 2–3x/week 💪"]

    return quests or ["8k steps/day 👟", "8 glasses of water 💧"]

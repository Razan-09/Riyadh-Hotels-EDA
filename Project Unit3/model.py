import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def make_features(rating, reviews, nights, source):
    """Turn raw values into the columns the model uses."""
    return pd.DataFrame({
        "rating": rating,
        "log_reviews": np.log1p(reviews),
        "nights": nights,
        "is_booking": (pd.Series(source) == "Booking.com").astype(int).values,
    })


def train(df):
    """Train a simple linear regression and test it on unseen data."""
    d = df.dropna(subset=["rating_10", "reviews_count"])

    X = make_features(d["rating_10"].values, d["reviews_count"].values,
                      d["stay_duration"].values, d["Source"].values)
    y = np.log1p(d["price_per_night"].values)  # log, because prices are very skewed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))  # R² on data the model has not seen
    return model, score


def predict_price(model, rating, reviews, nights, source):
    """Predict the price per night (in SAR) for one hotel."""
    X = make_features([rating], [reviews], [nights], [source])
    return float(np.expm1(model.predict(X)[0]))
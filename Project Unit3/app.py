import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

# ---------- Page setup ----------
st.set_page_config(page_title="Riyadh Hotels Dashboard", page_icon="🏨", layout="wide")

# Small style touch: card look for the metrics
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background-color: #F3F6F6;
        padding: 14px 18px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLOR = "#0F766E"


# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("Riyadh_Hotels_clean.csv", parse_dates=["checkin_date", "checkout_date"])
    # TripAdvisor rating is out of 5 and Booking.com is out of 10, so we put both on a 0-10 scale
    df["rating_10"] = df["rating"]
    df.loc[df["Source"] == "TripAdvisor", "rating_10"] = df["rating"] * 2
    return df


df = load_data()

# ---------- Sidebar ----------
st.sidebar.title("🏨 Riyadh Hotels")
st.sidebar.write(
    "Hotel offers in Riyadh (2026) collected from Booking.com and TripAdvisor, "
    "with price, rating, reviews and location. Prices are in SAR. "
    "These are displayed offers, not confirmed bookings."
)

st.sidebar.header("Filters")

# Dropdown
source = st.sidebar.selectbox("Source", ["All"] + sorted(df["Source"].unique()))

# Slider
max_price = int(df["price_per_night"].quantile(0.99))
price_range = st.sidebar.slider("Price per night (SAR)", 0, max_price, (0, max_price))

# Multiselect
stays = sorted(df["stay_duration"].unique())
selected_stays = st.sidebar.multiselect("Stay duration (nights)", stays, default=stays)

# Apply the filters
data = df[
    df["price_per_night"].between(price_range[0], price_range[1])
    & df["stay_duration"].isin(selected_stays)
]
if source != "All":
    data = data[data["Source"] == source]

# ---------- Main page ----------
st.title("Riyadh Hotels Insights Dashboard")
st.caption("Use the filters in the sidebar to explore the data.")

if data.empty:
    st.warning("No data for these filters. Try changing them.")
    st.stop()

# Key numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Offers", f"{len(data):,}")
c2.metric("Unique hotels", f"{data['hotel_id'].nunique():,}")
c3.metric("Median price / night", f"{data['price_per_night'].median():,.0f} SAR")
c4.metric("Average rating (0-10)", f"{data['rating_10'].mean():.1f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data", "Visualizations", "Insights", "Price predictor", "About"]
)

# ----- Tab 1: data -----
with tab1:
    st.subheader("Data preview")
    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("Summary statistics")
    cols = ["price_per_night", "rating_10", "reviews_count", "stay_duration"]
    st.dataframe(data[cols].describe().round(2), use_container_width=True)

# ----- Tab 2: charts -----
with tab2:
    left, right = st.columns(2)

    with left:
        st.subheader("Price per night")
        fig = px.histogram(data, x="price_per_night", nbins=50,
                           color_discrete_sequence=[COLOR])
        fig.update_xaxes(type="log", title="Price per night (SAR, log scale)")
        fig.update_layout(yaxis_title="Offers", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Price by source")
        fig = px.box(data, x="Source", y="price_per_night", color="Source",
                     color_discrete_sequence=[COLOR, "#F59E0B"])
        fig.update_yaxes(type="log", title="Price per night (SAR, log scale)")
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rating vs price")
    fig = px.scatter(data, x="rating_10", y="price_per_night", color="Source",
                     hover_name="hotel_name", opacity=0.5,
                     color_discrete_sequence=[COLOR, "#F59E0B"])
    fig.update_yaxes(type="log", title="Price per night (SAR, log scale)")
    fig.update_xaxes(title="Rating (0-10)")
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Median price by check-in date")
    daily = data.groupby(["checkin_date", "Source"])["price_per_night"].median().reset_index()
    fig = px.line(daily, x="checkin_date", y="price_per_night", color="Source",
                  color_discrete_sequence=[COLOR, "#F59E0B"])
    fig.update_layout(yaxis_title="Median price per night (SAR)", xaxis_title="Check-in date",
                      margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hotel locations")
    st.caption("Only points inside the Riyadh area are shown.")
    area = data[data["latitude"].between(24.2, 25.3) & data["longitude"].between(46.2, 47.2)]
    st.map(area[["latitude", "longitude"]])

# ----- Tab 3: insights -----
with tab3:
    st.subheader("Key insights")
    st.markdown(
        """
- **Prices are right-skewed.** The median nightly price is about 350 SAR and the mean about 463 SAR, because a few luxury hotels are very expensive.
- **Source matters.** Booking.com is much more expensive than TripAdvisor, and the two use different rating scales (10 vs 5).
- **Rating and price.** The overall correlation is 0.63, but inside each source it is weaker (0.36 Booking.com, 0.28 TripAdvisor).
- **Reviews.** More reviews are moderately related to higher price (0.49) and weakly to higher rating (0.32).
- **Stay duration.** Longer stays are only slightly cheaper per night (correlation -0.10).
- **Dates.** Prices rise a little at the end of May, but there is no confirmed holiday effect, and the weekday has no clear effect.
- **Location.** Hotels are concentrated in one dense area of Riyadh.
        """
    )
    st.info("These insights come from the full dataset, not from the filtered data.")

# ----- Tab 4: simple predictive model -----
with tab4:
    st.subheader("Simple price predictor")
    st.write("A simple linear regression that estimates the **price per night** from rating, reviews, stay length and source.")


    @st.cache_resource
    def train_model():
        d = df.dropna(subset=["rating_10", "reviews_count"]).copy()
        X = pd.DataFrame({
            "rating": d["rating_10"],
            "log_reviews": np.log1p(d["reviews_count"]),
            "nights": d["stay_duration"],
            "is_booking": (d["Source"] == "Booking.com").astype(int),
        })
        y = np.log1p(d["price_per_night"])  # log, because prices are very skewed
        model = LinearRegression().fit(X, y)
        return model, model.score(X, y)


    model, r2 = train_model()

    a, b = st.columns(2)
    with a:
        p_source = st.selectbox("Source", ["Booking.com", "TripAdvisor"], key="p_source")
        p_rating = st.slider("Rating (0-10)", 1.0, 10.0, 8.0, 0.1)
    with b:
        p_reviews = st.number_input("Number of reviews", min_value=0, value=100)
        p_nights = st.number_input("Nights", min_value=1, max_value=30, value=3)

    x_new = pd.DataFrame({
        "rating": [p_rating],
        "log_reviews": [np.log1p(p_reviews)],
        "nights": [p_nights],
        "is_booking": [int(p_source == "Booking.com")],
    })
    pred = np.expm1(model.predict(x_new)[0])

    st.metric("Estimated price per night", f"{pred:,.0f} SAR")
    st.caption(f"Model R² = {r2:.2f}. This is a simple model for learning, so use the estimate as a rough guide only.")

# ----- Tab 5: about -----
with tab5:
    st.subheader("About this project")
    st.write(
        "Exploratory data analysis of Riyadh hotel offers. The data was cleaned "
        "(duplicates removed, prices converted to SAR, price per night calculated) "
        "before building this dashboard."
    )
    st.write("**Limitations:** about one month of data, offers rather than real bookings, "
             "and some missing TripAdvisor prices were filled with the source median.")

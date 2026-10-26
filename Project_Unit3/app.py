from pathlib import Path
import base64
import pandas as pd
import plotly.express as px
import streamlit as st
from model import predict_price, train

HERE = Path(__file__).parent
COLORS = ["#0F766E", "#F59E0B"]  # Booking.com, TripAdvisor

# Page setup 
st.set_page_config(page_title="Riyadh Hotels Dashboard", layout="wide")

# Load the design from the CSS file
st.markdown(f"<style>{(HERE / 'style.css').read_text()}</style>", unsafe_allow_html=True)

logo_b64 = base64.b64encode((HERE / "logo.png").read_bytes()).decode()


def html(text):
    """Show HTML (removes indentation so Markdown does not treat it as code)."""
    st.markdown("\n".join(line.strip() for line in text.splitlines() if line.strip()), unsafe_allow_html=True)


def section(anchor, title, subtitle=""):
    html(f'<div id="{anchor}"></div><div class="section-title">{title}</div>'
         f'<div class="section-sub">{subtitle}</div>')


def style_chart(fig):
    fig.update_layout(template="plotly_white", margin=dict(t=20, b=10),
                      font=dict(family="Poppins, sans-serif"))
    return fig


#  data 
@st.cache_data
def load_data():
    df = pd.read_csv(HERE / "Riyadh_Hotels_clean.csv", parse_dates=["checkin_date", "checkout_date"])
    
    # TripAdvisor rating is out of 5 and Booking.com is out of 10, so we use a common 0-10 scale
    df["rating_10"] = df["rating"]
    df.loc[df["Source"] == "TripAdvisor", "rating_10"] = df["rating"] * 2
    return df


df = load_data()

# Sidebar
st.sidebar.title("Riyadh Hotels 2026")
st.sidebar.write(
    "Hotel offers in Riyadh (2026) collected from Booking.com and TripAdvisor, "
    "with price, rating, reviews and location. Prices are in SAR. "
    "These are displayed offers, not confirmed bookings."
)
st.sidebar.header("Filters")

source = st.sidebar.selectbox("Source", ["All"] + sorted(df["Source"].unique()))

max_price = int(df["price_per_night"].quantile(0.99))
price_range = st.sidebar.slider("Price per night (SAR)", 0, max_price, (0, max_price))

stays = sorted(df["stay_duration"].unique())
selected_stays = st.sidebar.multiselect("Stay duration (nights)", stays, default=stays)

data = df[
    df["price_per_night"].between(price_range[0], price_range[1])
    & df["stay_duration"].isin(selected_stays)
]
if source != "All":
    data = data[data["Source"] == source]

# Header with logo 
html(f"""
<div class="hero">
<div class="logo"><img src="data:image/png;base64,{logo_b64}" alt="logo"></div>
<div>
<h1>Riyadh Hotels Insights</h1>
<p>Prices, ratings and locations of hotel offers in Riyadh, 2026</p>
</div>
</div>
<div class="nav">
<a href="#summary" target="_self">Summary</a>
<a href="#data" target="_self">Data</a>
<a href="#charts" target="_self">Charts</a>
<a href="#insights" target="_self">Insights</a>
<a href="#predictor" target="_self">Predictor</a>
</div>
""")

# 1. Data summary (before the data) 
section("summary", "Data summary", "A quick overview of the whole dataset before looking at the rows")

n_rows, n_hotels = len(df), df["hotel_id"].nunique()
first, last = df["checkin_date"].min().date(), df["checkin_date"].max().date()
booking_share = (df["Source"] == "Booking.com").mean() * 100
no_rating = df["rating"].isna().mean() * 100

html(f"""
<div class="summary-box">
This dataset has <b>{n_rows:,}</b> hotel offers for <b>{n_hotels:,}</b> unique hotels in Riyadh,
with check-in dates from <b>{first}</b> to <b>{last}</b>.
About <b>{booking_share:.0f}%</b> of the offers come from Booking.com and the rest from TripAdvisor.
Each offer has the hotel location, rating, number of reviews and the price of the stay (converted to SAR),
and we added the price per night. About <b>{no_rating:.0f}%</b> of the offers have no rating
(hotels without reviews), and the same hotel can appear several times with different dates.
</div>
""")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Rows", f"{n_rows:,}")
s2.metric("Columns", f"{df.shape[1]}")
s3.metric("Unique hotels", f"{n_hotels:,}")
s4.metric("Date range (days)", f"{(df['checkin_date'].max() - df['checkin_date'].min()).days}")

descriptions = {
    "hotel_id": "Hotel identifier",
    "hotel_name": "Name of the hotel",
    "city": "City (all Riyadh)",
    "latitude": "Latitude of the hotel",
    "longitude": "Longitude of the hotel",
    "rating": "Original rating (TripAdvisor out of 5, Booking.com out of 10)",
    "reviews_count": "Number of reviews",
    "price_SAR": "Total price of the stay in SAR",
    "checkin_date": "Check-in date",
    "checkout_date": "Check-out date",
    "Source": "Where the offer comes from",
    "stay_duration": "Number of nights",
    "price_per_night": "Price per night in SAR",
    "rating_10": "Rating on a common 0-10 scale",
}
cols_table = pd.DataFrame({
    "Column": [c for c in df.columns if c in descriptions],
    "Description": [descriptions[c] for c in df.columns if c in descriptions],
})
st.write("**Columns**")
st.dataframe(cols_table, use_container_width=True, hide_index=True)

# 2. Data
section("data", "Data", "The tables below follow the filters in the sidebar")

if data.empty:
    st.warning("No data for these filters. Try changing them.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Offers", f"{len(data):,}")
c2.metric("Unique hotels", f"{data['hotel_id'].nunique():,}")
c3.metric("Median price / night", f"{data['price_per_night'].median():,.0f} SAR")
c4.metric("Average rating (0-10)", f"{data['rating_10'].mean():.1f}")

st.write("**Data preview**")
st.dataframe(data.head(20), use_container_width=True)

st.write("**Summary statistics**")
stat_cols = ["price_per_night", "rating_10", "reviews_count", "stay_duration"]
st.dataframe(data[stat_cols].describe().round(2), use_container_width=True)

# 3. Charts
section("charts", "Interactive charts", "Hover, zoom and click the legend to explore")

left, right = st.columns(2)
with left:
    st.write("**Price per night**")
    fig = px.histogram(data, x="price_per_night", nbins=50, color_discrete_sequence=[COLORS[0]])
    fig.update_xaxes(type="log", title="Price per night (SAR, log scale)")
    fig.update_layout(yaxis_title="Offers")
    st.plotly_chart(style_chart(fig), use_container_width=True)

with right:
    st.write("**Price by source**")
    fig = px.box(data, x="Source", y="price_per_night", color="Source", color_discrete_sequence=COLORS)
    fig.update_yaxes(type="log", title="Price per night (SAR, log scale)")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_chart(fig), use_container_width=True)

st.write("**Rating vs price**")
fig = px.scatter(data, x="rating_10", y="price_per_night", color="Source",
                 hover_name="hotel_name", opacity=0.5, color_discrete_sequence=COLORS)
fig.update_yaxes(type="log", title="Price per night (SAR, log scale)")
fig.update_xaxes(title="Rating (0-10)")
st.plotly_chart(style_chart(fig), use_container_width=True)

st.write("**Median price by check-in date**")
daily = data.groupby(["checkin_date", "Source"])["price_per_night"].median().reset_index()
fig = px.line(daily, x="checkin_date", y="price_per_night", color="Source", color_discrete_sequence=COLORS)
fig.update_layout(yaxis_title="Median price per night (SAR)", xaxis_title="Check-in date")
st.plotly_chart(style_chart(fig), use_container_width=True)

st.write("**Hotel locations** (only points inside the Riyadh area)")
area = data[data["latitude"].between(24.2, 25.3) & data["longitude"].between(46.2, 47.2)]
st.map(area[["latitude", "longitude"]], color="#0F766E")

# 4. Insights 
section("insights", "Key insights", "From the analysis of the full dataset (not affected by the filters)")

insights = [
    ("Prices", "Prices are right-skewed: the median is about 350 SAR and the mean about 463 SAR, because a few luxury hotels are very expensive."),
    ("Sources", "Booking.com is much more expensive than TripAdvisor, and the two use different rating scales (10 vs 5)."),
    ("Rating", "The overall correlation between rating and price is 0.63, but inside each source it is weaker (0.36 Booking.com, 0.28 TripAdvisor)."),
    ("Reviews", "More reviews are moderately related to higher price (0.49) and weakly to higher rating (0.32)."),
    ("Stay length", "Longer stays are only slightly cheaper per night (correlation -0.09)."),
    ("Dates", "Prices rise a little at the end of May, but there is no confirmed holiday effect and no clear weekday effect."),
    ("Outliers", "The most expensive offers are real luxury hotels and chalets. One offer is around 50,000 SAR per night and is treated as an extreme case."),
    ("Location", "Hotels are concentrated in one dense area of Riyadh. A few coordinates are far from the city and are probably wrong."),
]
cards = "".join(f'<div class="insight-card"><span class="tag">{t}</span><p>{text}</p></div>'
                for t, text in insights)
html(f'<div class="insight-grid">{cards}</div>')

# 5. Predictor 
section("predictor", "Price predictor", "A simple linear regression model for learning")
 
 
@st.cache_resource
def train_model():
    return train(df)
 
 
model, r2 = train_model()
 
a, b, c = st.columns(3)
with a:
    p_source = st.selectbox("Source", ["Booking.com", "TripAdvisor"], key="p_source")
    p_rating = st.slider("Rating (0-10)", 1.0, 10.0, 8.0, 0.1)
with b:
    p_reviews = st.number_input("Number of reviews", min_value=0, value=100)
    p_nights = st.number_input("Nights", min_value=1, max_value=30, value=3)
 
pred = predict_price(model, p_rating, p_reviews, p_nights, p_source)
 
with c:
    st.metric("Estimated price per night", f"{pred:,.0f} SAR")
    st.caption(f"Model R² on test data = {r2:.2f}. A simple model, so use the estimate as a rough guide only.")

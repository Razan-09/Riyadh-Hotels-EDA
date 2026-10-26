# Riyadh Hotels EDA (2026)

An exploratory data analysis project on hotel offers in Riyadh, Saudi Arabia, featuring an interactive Streamlit dashboard and a simple machine learning model for price prediction.

**Live Dashboard:** [Explore the app](https://riyadh-hotels-eda-sdkagcetapd2iuwkxupmc2.streamlit.app/)

## Project Overview

This project analyzes hotel offers in Riyadh to explore price distributions, ratings, customer reviews, stay duration, booking platforms, check-in dates, and geographical locations.

It combines exploratory data analysis (EDA), interactive data visualization, and a basic machine learning model to estimate hotel prices per night.

## Dataset

- **Source:** [Riyadh Hotels 2026 – Kaggle](https://www.kaggle.com/datasets/mohammedalsubaie/riyadh-hotels-2026)
- **Platforms:** Booking.com and TripAdvisor
- **Size:** Approximately 9,000 offers for 2,700 unique hotels after cleaning
- **Check-in date range:** April 27 – May 30, 2026
- **Target variable:** Price per night in SAR

Each row represents a hotel offer for a particular stay period. A hotel can appear multiple times with different dates and prices.

The dataset includes hotel names, geographical coordinates, ratings, review counts, total stay prices, check-in and check-out dates, and the source platform.

Prices represent displayed offers rather than confirmed bookings.

## Research Questions

The analysis investigates the following questions:

1. How are nightly hotel prices distributed, and what is a typical price?
2. How are hotel ratings and review counts associated with nightly prices?
3. How does stay duration relate to the price per night?
4. How do prices vary across check-in dates?
5. How do Booking.com and TripAdvisor offers compare?
6. Where are the available hotel offers located in Riyadh?
7. Can a simple regression model estimate the price per night?

## Data Cleaning and Feature Engineering

The main preprocessing steps included:

- Removing duplicate records.
- Filling missing city values with Riyadh.
- Removing records without valid geographical coordinates.
- Replacing zero ratings with missing values when they represented unrated hotels.
- Converting TripAdvisor prices from USD to SAR using a conversion multiplier of 3.75.
- Calculating stay duration from check-in and check-out dates.
- Calculating price per night by dividing the total stay price by the number of nights.
- Filling missing prices using the median price for each source.
- Retaining extreme prices that may represent genuine luxury offers, while using medians and logarithmic scales to help interpret skewed distributions.
- Standardizing ratings to a common 0–10 scale for comparison and modeling.

The original rating scales differ between platforms: Booking.com ratings are out of 10, while TripAdvisor ratings are out of 5. TripAdvisor ratings were multiplied by two to create the `rating_10` feature.

## Key Findings

The dashboard highlights several patterns in the cleaned dataset:

- **Price distribution:** Nightly prices are right-skewed. The median is more representative of a typical offer than the mean when a small number of expensive offers increase the average.
- **Price and rating:** The dashboard reports a positive association between hotel ratings and nightly prices. The strength of this relationship varies by platform.
- **Price and reviews:** The reported correlation between nightly price and review count is approximately 0.49, suggesting a moderate positive association.
- **Rating and reviews:** The reported correlation is approximately 0.32, indicating a weak positive association.
- **Stay duration:** The reported correlation between stay duration and nightly price is approximately -0.09, indicating a very weak negative relationship.
- **Dates and location:** The dashboard provides a daily price comparison and a geographical map to explore changes over time and the distribution of hotel offers.

These findings describe associations in the available data. They do not establish that ratings, reviews, dates, or platform choice directly cause price differences.

*Note: Verify all reported statistics against the final cleaned dataset before treating them as definitive results.*

## Interactive Streamlit Dashboard

The dashboard includes the following features:

- **Dataset summary:** Total offers, unique hotels, date range, source distribution, and missing ratings.
- **Interactive filters:** Filter by source, price per night, and stay duration.
- **Data exploration:** Preview the filtered data and view descriptive statistics.
- **Interactive visualizations:**
  - Nightly price distribution using a histogram and logarithmic scale.
  - Price comparison between Booking.com and TripAdvisor using boxplots.
  - Rating versus nightly price scatterplot.
  - Median nightly price by check-in date.
  - Geographical map of hotel offers within the selected Riyadh area.
- **Key insights:** A summary of the project's main observations.
- **Price predictor:** Estimate the nightly price based on selected hotel characteristics.

**Try it here:** [Riyadh Hotels Insights](https://riyadh-hotels-eda-a4wsjwkqrmva2shpkyxp9n.streamlit.app)

## Machine Learning: Price Prediction

A **Linear Regression** model was implemented to estimate hotel prices per night.

### Features

| Feature | Description |
|---|---|
| `rating` | Hotel rating on a 0–10 scale |
| `log_reviews` | Log-transformed number of reviews |
| `nights` | Duration of the stay |
| `is_booking` | Binary indicator for Booking.com offers |

The target variable is the logarithm of the price per night in SAR.

### Training and Evaluation

- **Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Train-test split:** 80% training and 20% testing
- **Random state:** 42
- **Evaluation metric:** R² score on the test set
- **Target transformation:** `log1p` to reduce the impact of right-skewed prices
- **Prediction transformation:** `expm1` to convert predictions back to SAR

The dashboard displays the model's test-set R² score and allows users to enter hotel characteristics to obtain an estimated nightly price.

The model is intended as a simple demonstration of regression rather than a definitive hotel-pricing system. Its predictions should be interpreted as approximate estimates.

## Tools and Technologies

- **Python** – programming and analysis
- **Pandas and NumPy** – data cleaning, transformation, and feature engineering
- **Matplotlib and Seaborn** – exploratory visualization
- **Plotly Express** – interactive charts
- **Scikit-learn** – linear regression and model evaluation
- **Streamlit** – interactive web application
- **CSS and HTML** – dashboard styling and layout
- **GitHub** – source code management and deployment workflow

## Limitations

- The dataset covers a limited date range and cannot establish annual or seasonal pricing patterns.
- Displayed prices are not necessarily final transaction prices.
- Platform comparisons may be affected by differences in the hotels and offers listed by each source.
- Median imputation can influence the distribution of prices.
- A random row-based train-test split may place offers from the same hotel in both sets, potentially making model performance appear more favorable.
- The model uses a limited number of features and does not include all factors that may affect hotel prices.
- The map shows hotel coordinates, not verified distances to metro stations or event venues.

## Future Improvements

- Extend the dataset to cover more dates and seasons.
- Compare additional regression models and report MAE, RMSE, and R².
- Evaluate performance using a hotel-based split to test how well the model generalizes to unseen hotels.
- Explore additional features such as hotel category and distance to major landmarks, where reliable data is available.
- Improve price prediction through additional relevant features and model validation.
- Add further interactive comparisons and filters to the dashboard.

## Conclusion

This project demonstrates a practical data-analysis workflow, from cleaning and feature engineering to exploratory visualization, an interactive dashboard, and basic machine learning. It provides a structured way to explore Riyadh hotel offers and investigate the factors associated with their nightly prices.

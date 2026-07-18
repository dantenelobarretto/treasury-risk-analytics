import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def forecast_treasury_risk(
    csv_path,
    variable="net_exposure_dc",
    entity=None,
    forecast_days=30,
    arima_order=None,  # If None, auto-selects d based on ADF test
    plot=True,
) -> pd.DataFrame:
    """
    Forecast Treasury Risk Using ARIMA
    
    Parameters
    ----------
    csv_path : str
        Path to synthetic_treasury_risk_data.csv
    variable : str
        Treasury risk variable to forecast. Default: net_exposure_dc
    entity : str
        Optional entity filter
    forecast_days : int
        Forecast horizon in days
    arima_order : tuple or None
        ARIMA(p,d,q) order. If None, d is determined via ADF stationarity test.
    plot : bool
        Whether to render the forecast chart
    
    Returns
    -------
    pd.DataFrame
        forecast_mean, lower_ci, upper_ci
    """

    df = pd.read_csv(csv_path)
    df["valuation_date"] = pd.to_datetime(df["valuation_date"])

    if entity:
        df = df[df["legal_entity"] == entity]
        if df.empty:
            raise ValueError(f"No data found for entity '{entity}'.")
        
    # Aggregate to daily time series
    ts = (
        df.groupby("valuation_date")[variable]
        .sum()
        .sort_index()
        .asfreq("D")        # enforce daily frequency so ARIMA sees no gaps
        .ffill()            # forward-fill any missing dates
    )

    # Determine differencing order via ADF test
    if arima_order is None:
        adf_result = adfuller(ts.dropna())
        adf_pvalue = adf_result[1]

        if adf_pvalue < 0.05:
            # Already stationary — no differencing needed
            d = 0
        else:
            # Non-stationary — one round of differencing
            d = 1

        arima_order = (2, d, 2)
        print(f"ADF p-value: {adf_pvalue:.4f} → using ARIMA{arima_order}")
    else:
        print(f"Using provided ARIMA{arima_order}")

    # Fit ARIMA
    model = ARIMA(ts, order=arima_order)
    fitted = model.fit()

    # Forecast with confidence intervals
    forecast_obj = fitted.get_forecast(steps=forecast_days)
    forecast_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.05)    # 95% CI

    future_dates = pd.date_range(
        start=ts.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D",
    )

    forecast_mean.index = future_dates
    conf_int.index = future_dates

    result = pd.DataFrame({
        "forecast_mean": forecast_mean,
        "lower_ci":      conf_int.iloc[:, 0],
        "upper_ci":      conf_int.iloc[:, 1],
    })

    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(ts, label="Historical Exposure", color="#1f77b4")

        ax.plot(
            result["forecast_mean"],
            label="Forecast Exposure",
            linestyle="--",
            color="#ff7f0e",
        )

        ax.fill_between(
            result.index,
            result["lower_ci"],
            result["upper_ci"],
            alpha=0.25,
            color="#ff7f0e",
            label="95% Confidence Interval",
        )

        title = f"Treasury Net FX Exposure Forecast (ARIMA{arima_order})"
        if entity:
            title += f" — {entity}"

        ax.set_title(title)
        ax.set_ylabel("Net Exposure (Domestic Currency)")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    print(fitted.summary())
    return result
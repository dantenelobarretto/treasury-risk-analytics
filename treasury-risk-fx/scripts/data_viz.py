import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_treasury_dashboard(
    csv_path,
    entity=None,
    currencies=None,
    start_date=None,
    end_date=None,
    figsize=(20, 16)
) -> plt.figure:
    """
    Advanced Treasury Risk Dashboard (8 Charts)

    Designed for corporate treasury / bank FX risk monitoring.

    Panels
    ------
    1 FX Spot Rates
    2 FX Volatility
    3 Gross vs Net Exposure
    4 Hedge Coverage vs Policy
    5 Value-at-Risk vs Stress Loss
    6 Hedge MTM PnL
    7 Currency Exposure Concentration
    8 Exposure Maturity Ladder
    """

    sns.set_style("whitegrid")

    df = pd.read_csv(csv_path)
    df["valuation_date"] = pd.to_datetime(df["valuation_date"])

    # ----------------------------
    # Filters
    # ----------------------------

    if entity:
        df = df[df["legal_entity"] == entity]

    if currencies:
        df = df[df["currency"].isin(currencies)]

    if start_date:
        df = df[df["valuation_date"] >= start_date]

    if end_date:
        df = df[df["valuation_date"] <= end_date]

    # ----------------------------
    # Aggregations
    # ----------------------------

    daily_currency = (
        df.groupby(["valuation_date", "currency"])
        .mean(numeric_only=True)
        .reset_index()
    )

    exposure_daily = (
        df.groupby("valuation_date")
        .sum(numeric_only=True)
        .reset_index()
    )

    coverage_daily = (
        df.groupby("valuation_date")
        .mean(numeric_only=True)
        .reset_index()
    )

    mtm_daily = (
        df.groupby("valuation_date")["mtm_hedge"]
        .sum()
        .reset_index()
    )

    exposure_currency = (
        df.groupby("currency")["net_exposure_dc"]
        .sum()
        .reset_index()
    )

    maturity_ladder = (
        df.groupby(["maturity_bucket"])["net_exposure_dc"]
        .sum()
        .reset_index()
    )

    # ----------------------------
    # Layout
    # ----------------------------

    fig, axes = plt.subplots(4, 2, figsize=figsize)

    # ----------------------------
    # 1 FX Spot Rates
    # ----------------------------

    sns.lineplot(
        data=daily_currency,
        x="valuation_date",
        y="spot_rate",
        hue="currency",
        ax=axes[0, 0]
    )

    axes[0, 0].set_title("FX Spot Rate Dynamics")

    # ----------------------------
    # 2 FX Volatility
    # ----------------------------

    sns.lineplot(
        data=daily_currency,
        x="valuation_date",
        y="fx_volatility",
        hue="currency",
        ax=axes[0, 1]
    )

    axes[0, 1].set_title("FX Volatility (GARCH)")

    # ----------------------------
    # 3 Gross vs Net Exposure
    # ----------------------------

    sns.lineplot(
        data=exposure_daily,
        x="valuation_date",
        y="gross_exposure_fc",
        ax=axes[1, 0],
        label="Gross Exposure"
    )

    sns.lineplot(
        data=exposure_daily,
        x="valuation_date",
        y="net_exposure_fc",
        ax=axes[1, 0],
        label="Net Exposure"
    )

    axes[1, 0].set_title("Gross vs Net FX Exposure")

    # ----------------------------
    # 4 Hedge Coverage
    # ----------------------------

    sns.lineplot(
        data=coverage_daily,
        x="valuation_date",
        y="coverage_ratio",
        ax=axes[1, 1],
        label="Actual Coverage"
    )

    sns.lineplot(
        data=coverage_daily,
        x="valuation_date",
        y="hedge_ratio_policy",
        ax=axes[1, 1],
        label="Policy Target"
    )

    axes[1, 1].set_title("Hedge Coverage vs Policy")

    # ----------------------------
    # 5 VaR vs Stress Loss
    # ----------------------------

    sns.lineplot(
        data=coverage_daily,
        x="valuation_date",
        y="VaR_95",
        ax=axes[2, 0],
        label="VaR 95%"
    )

    sns.lineplot(
        data=coverage_daily,
        x="valuation_date",
        y="stress_loss_10pct",
        ax=axes[2, 0],
        label="Stress Loss (-10%)"
    )

    axes[2, 0].set_title("Treasury Risk Metrics")

    # ----------------------------
    # 6 Hedge MTM PnL
    # ----------------------------

    sns.lineplot(
        data=mtm_daily,
        x="valuation_date",
        y="mtm_hedge",
        ax=axes[2, 1]
    )

    axes[2, 1].set_title("Hedge Mark-to-Market PnL")

    # ----------------------------
    # 7 Currency Exposure
    # ----------------------------

    sns.barplot(
        data=exposure_currency,
        x="currency",
        y="net_exposure_dc",
        ax=axes[3, 0]
    )

    axes[3, 0].set_title("Currency Exposure Concentration")

    # ----------------------------
    # 8 Maturity Ladder
    # ----------------------------

    sns.barplot(
        data=maturity_ladder,
        x="maturity_bucket",
        y="net_exposure_dc",
        ax=axes[3, 1]
    )

    axes[3, 1].set_title("Exposure by Maturity Bucket")

    plt.tight_layout()

    return fig
"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=self.price.index, columns=self.price.columns)

        """
        TODO: Complete Task 4 Below
        """
        prices = self.price[assets]
        returns = self.returns[assets]
        spy_prices = self.price[self.exclude]
        spy_returns = self.returns[self.exclude]

        # Trend filters (short and long) for more robust regime detection
        short_window = self.lookback
        long_window = max(int(self.lookback * 4), self.lookback + 1)  # e.g. 50 -> 200

        ma_short_assets = prices.rolling(short_window).mean()
        ma_long_assets = prices.rolling(long_window).mean()

        ma_short_spy = spy_prices.rolling(short_window).mean()
        ma_long_spy = spy_prices.rolling(long_window).mean()

        # Rolling volatility for inverse-vol weighting (use 63 trading days ~ 3 months)
        vol_window = 63
        rolling_vol = returns.rolling(vol_window).std()

        # SPY volatility for simple volatility targeting
        spy_rolling_vol = spy_returns.rolling(vol_window).std()
        target_spy_vol = spy_rolling_vol.median()  # long-run typical SPY volatility

        for date in self.price.index:
            # Skip until we have enough history for both MAs and volatility
            if (
                date not in ma_short_assets.index
                or date not in ma_long_assets.index
                or date not in rolling_vol.index
            ):
                continue

            # Time-series trend filter: asset and SPY both above their short & long MAs
            asset_above_ma = (prices.loc[date] > ma_short_assets.loc[date]) & (
                prices.loc[date] > ma_long_assets.loc[date]
            )
            spy_above_ma = (spy_prices.loc[date] > ma_short_spy.loc[date]) and (
                spy_prices.loc[date] > ma_long_spy.loc[date]
            )

            if not spy_above_ma:
                # If the broad market (SPY) is below trend, explicitly go to cash
                # on this date so that we do not carry over old positions.
                self.portfolio_weights.loc[date, assets] = 0.0
                continue

            # Select assets that are in uptrend
            active = asset_above_ma[asset_above_ma].index.tolist()

            if len(active) == 0:
                # No active assets -> remain in cash
                continue

            # Inverse-vol allocation among active assets
            vol_today = rolling_vol.loc[date, active].replace(0, np.nan)
            inv_vol = 1 / vol_today
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()

            if inv_vol.empty:
                continue

            weights = inv_vol / inv_vol.sum()

            # Cap single-asset risk contribution for diversification (e.g. <= 30%)
            max_weight = 0.3
            weights = weights.clip(upper=max_weight)
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()

            # Simple volatility targeting based on SPY to reduce risk in very volatile regimes
            spy_vol_today = spy_rolling_vol.loc[date]
            if pd.isna(spy_vol_today) or pd.isna(target_spy_vol) or target_spy_vol == 0:
                exposure = 1.0
            else:
                # When SPY is more volatile than usual, scale down gross exposure
                exposure = min(1.0, float(target_spy_vol / (spy_vol_today + 1e-8)))

            # First clear any existing allocations on this date to avoid leverage
            self.portfolio_weights.loc[date, assets] = 0.0

            # Assign weights on this date (SPY column stays NaN/0)
            self.portfolio_weights.loc[date, active] = (exposure * weights.values)

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)

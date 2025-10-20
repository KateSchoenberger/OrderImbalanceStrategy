Order Imbalance Strategy - Machine Learning Implementation

A supervised learning strategy inspired by Darryl Shen’s research on order imbalance, re-implemented in Python using modern machine learning classifiers (Logistic Regression, SVM, Random Forest).
This project detects short-term order imbalance signals using microstructure-based features — MPB (Market Pressure Balance), VOI (Volume Order Imbalance), and OIR (Order Imbalance Ratio) — achieving >90% accuracy in classification on test data.

# Overview

This repository contains a modular Python framework for detecting and backtesting order-imbalance signals using ML.
It includes:
Feature computation (MPB, VOI, OIR, rolling statistics)
Supervised model training (Logistic Regression, SVM, Random Forest, Linear Regression)
Chronological cross-validation to avoid lookahead bias
Simple backtesting engine for performance visualization

Features Explained (3):
MPB (Market Pressure Balance)	(buy_volume - sell_volume) / total_volume — measures directional volume pressure
VOI (Volume Order Imbalance)	Rolling ratio of signed to total volume — captures cumulative buy/sell imbalance
OIR (Order Imbalance Ratio)	log((aggressive_buy_count + ε) / (aggressive_sell_count + ε)) — measures aggressiveness asymmetry

These features were inspired by Darryl Shen’s research on high-frequency order flow and imbalance modeling.


# Features

High Accuracy: Achieves over 90% classification accuracy on test data.

Modular Design: Easily extendable for different classifiers and features.

Backtesting Framework: Assess strategy performance on historical data.

# Installation

To set up the project on your local machine:

Clone the repository:

git clone https://github.com/KateSchoenberger/OrderImbalanceStrategy.git
cd OrderImbalanceStrategy


Install the required dependencies:

pip install -r requirements.txt


Prepare your market data in a suitable format for feature computation.

# Usage

Compute Features: Use the provided scripts to calculate MPB, VOI, and OIR from your market data.

Train Models: Apply the classifiers to the computed features to train your models.

Evaluate Performance: Use the backtesting framework to assess the strategy's performance on historical data.

# Testing

To run the test suite:

pytest tests/


Ensure all tests pass before deploying to production.

# Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your proposed changes.

# License

This project is licensed under the MIT License - see the LICENSE
 file for details.

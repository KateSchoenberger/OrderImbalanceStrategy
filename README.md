Order Imbalance Strategy - Machine Learning Implementation

A supervised learning strategy inspired by Darryl Shen’s research on order imbalance, re-implemented in Python using modern machine learning classifiers (Logistic Regression, SVM, Random Forest).
This project detects short-term order imbalance signals using microstructure-based features — MPB (Market Pressure Balance), VOI (Volume Order Imbalance), and OIR (Order Imbalance Ratio) — achieving >90% accuracy in classification on test data.

Overview

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

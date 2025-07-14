"""
Model Validation and Testing Framework
Comprehensive validation, cross-validation, and statistical testing for trading models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest, anderson
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf
from arch import arch_model
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera as jb_test
from statsmodels.tsa.stattools import adfuller, kpss
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

@dataclass
class ValidationResult:
    """Individual validation test result"""
    test_name: str
    test_type: str  # 'statistical', 'cross_validation', 'robustness', 'performance'
    passed: bool
    score: float
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ModelValidationReport:
    """Comprehensive model validation report"""
    model_name: str
    validation_date: datetime
    overall_score: float
    overall_status: str  # 'PASS', 'FAIL', 'WARNING'
    
    # Test results by category
    statistical_tests: List[ValidationResult]
    cross_validation_tests: List[ValidationResult]
    robustness_tests: List[ValidationResult]
    performance_tests: List[ValidationResult]
    
    # Summary statistics
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    
    # Recommendations
    critical_issues: List[str]
    improvement_suggestions: List[str]
    deployment_recommendation: str

class ModelValidationFramework:
    """Comprehensive model validation and testing framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.test_results = {}
        self.validation_history = []
        
        # Test thresholds
        self.thresholds = {
            'min_r2': 0.1,
            'max_mse': 1.0,
            'min_sharpe': 0.5,
            'max_drawdown': 0.2,
            'min_stability': 0.7,
            'max_overfitting': 0.3,
            'min_statistical_power': 0.8,
            'alpha_level': 0.05
        }
        
        print("🔬 Model Validation Framework initialized")
        print(f"   Test categories: {len(self.config['test_categories'])}")
        print(f"   Validation methods: {len(self.config['validation_methods'])}")
    
    def _default_config(self):
        return {
            'test_categories': [
                'statistical_tests',
                'cross_validation',
                'robustness_tests',
                'performance_tests'
            ],
            'validation_methods': [
                'time_series_split',
                'walk_forward',
                'monte_carlo',
                'bootstrap'
            ],
            'statistical_tests': [
                'normality',
                'stationarity',
                'autocorrelation',
                'heteroscedasticity',
                'model_adequacy'
            ],
            'robustness_tests': [
                'noise_sensitivity',
                'parameter_stability',
                'data_snooping',
                'regime_change',
                'outlier_sensitivity'
            ],
            'performance_metrics': [
                'accuracy',
                'precision',
                'recall',
                'f1_score',
                'sharpe_ratio',
                'information_ratio',
                'maximum_drawdown'
            ],
            'cross_validation': {
                'n_splits': 5,
                'test_size': 0.2,
                'gap': 0,
                'max_train_size': None
            }
        }
    
    def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                      model_name: str, model_type: str = 'regression') -> ModelValidationReport:
        """Comprehensive model validation"""
        print(f"🔬 Starting comprehensive validation for {model_name}")
        print(f"   Model type: {model_type}")
        print(f"   Data shape: {X.shape}")
        
        validation_start = time.time()
        
        # Initialize results
        all_results = {
            'statistical_tests': [],
            'cross_validation_tests': [],
            'robustness_tests': [],
            'performance_tests': []
        }
        
        try:
            # 1. Statistical Tests
            print("📊 Running statistical tests...")
            all_results['statistical_tests'] = self._run_statistical_tests(model, X, y, model_type)
            
            # 2. Cross-Validation Tests
            print("🔄 Running cross-validation tests...")
            all_results['cross_validation_tests'] = self._run_cross_validation_tests(model, X, y, model_type)
            
            # 3. Robustness Tests
            print("🛡️  Running robustness tests...")
            all_results['robustness_tests'] = self._run_robustness_tests(model, X, y, model_type)
            
            # 4. Performance Tests
            print("📈 Running performance tests...")
            all_results['performance_tests'] = self._run_performance_tests(model, X, y, model_type)
            
            # Calculate overall score and status
            overall_score, overall_status = self._calculate_overall_assessment(all_results)
            
            # Generate recommendations
            critical_issues, improvement_suggestions, deployment_rec = self._generate_recommendations(all_results)
            
            # Create comprehensive report
            report = ModelValidationReport(
                model_name=model_name,
                validation_date=datetime.now(),
                overall_score=overall_score,
                overall_status=overall_status,
                statistical_tests=all_results['statistical_tests'],
                cross_validation_tests=all_results['cross_validation_tests'],
                robustness_tests=all_results['robustness_tests'],
                performance_tests=all_results['performance_tests'],
                total_tests=sum(len(results) for results in all_results.values()),
                passed_tests=sum(sum(1 for r in results if r.passed) for results in all_results.values()),
                failed_tests=sum(sum(1 for r in results if not r.passed) for results in all_results.values()),
                warning_tests=0,  # Could be expanded
                critical_issues=critical_issues,
                improvement_suggestions=improvement_suggestions,
                deployment_recommendation=deployment_rec
            )
            
            # Store results
            self.test_results[model_name] = report
            self.validation_history.append(report)
            
            validation_time = time.time() - validation_start
            print(f"✅ Validation completed in {validation_time:.2f} seconds")
            print(f"   Overall Score: {overall_score:.2f}")
            print(f"   Status: {overall_status}")
            print(f"   Tests Passed: {report.passed_tests}/{report.total_tests}")
            
            return report
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return self._create_failed_report(model_name, str(e))
    
    def _run_statistical_tests(self, model: Any, X: np.ndarray, y: np.ndarray, 
                              model_type: str) -> List[ValidationResult]:
        """Run comprehensive statistical tests"""
        results = []
        
        try:
            # Make predictions for residual analysis
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
                residuals = y - y_pred
            else:
                residuals = None
                y_pred = None
            
            # 1. Normality Tests
            if residuals is not None:
                results.extend(self._test_normality(residuals))
            
            # 2. Stationarity Tests
            results.extend(self._test_stationarity(y))
            
            # 3. Autocorrelation Tests
            if residuals is not None:
                results.extend(self._test_autocorrelation(residuals))
            
            # 4. Heteroscedasticity Tests
            if residuals is not None and y_pred is not None:
                results.extend(self._test_heteroscedasticity(residuals, y_pred))
            
            # 5. Model Adequacy Tests
            if residuals is not None:
                results.extend(self._test_model_adequacy(y, y_pred, residuals))
            
        except Exception as e:
            print(f"⚠️  Error in statistical tests: {e}")
            results.append(ValidationResult(
                test_name="statistical_tests_error",
                test_type="statistical",
                passed=False,
                score=0.0,
                metadata={"error": str(e)}
            ))
        
        return results
    
    def _test_normality(self, residuals: np.ndarray) -> List[ValidationResult]:
        """Test normality of residuals"""
        results = []
        
        # Shapiro-Wilk test (for small samples)
        if len(residuals) <= 5000:
            stat, p_value = shapiro(residuals)
            results.append(ValidationResult(
                test_name="shapiro_wilk_normality",
                test_type="statistical",
                passed=p_value > self.thresholds['alpha_level'],
                score=p_value,
                p_value=p_value,
                metadata={"statistic": stat, "sample_size": len(residuals)},
                recommendations=["Residuals should be normally distributed"] if p_value <= self.thresholds['alpha_level'] else []
            ))
        
        # Jarque-Bera test
        stat, p_value = jarque_bera(residuals)
        results.append(ValidationResult(
            test_name="jarque_bera_normality",
            test_type="statistical",
            passed=p_value > self.thresholds['alpha_level'],
            score=p_value,
            p_value=p_value,
            metadata={"statistic": stat},
            recommendations=["Consider data transformation or different model"] if p_value <= self.thresholds['alpha_level'] else []
        ))
        
        # Anderson-Darling test
        result = anderson(residuals, dist='norm')
        critical_value = result.critical_values[2]  # 5% significance
        results.append(ValidationResult(
            test_name="anderson_darling_normality",
            test_type="statistical",
            passed=result.statistic < critical_value,
            score=critical_value - result.statistic,
            metadata={"statistic": result.statistic, "critical_value": critical_value},
            recommendations=["Residuals deviate from normality"] if result.statistic >= critical_value else []
        ))
        
        return results
    
    def _test_stationarity(self, series: np.ndarray) -> List[ValidationResult]:
        """Test stationarity of time series"""
        results = []
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series)
        results.append(ValidationResult(
            test_name="adf_stationarity",
            test_type="statistical",
            passed=adf_result[1] < self.thresholds['alpha_level'],
            score=1 - adf_result[1],
            p_value=adf_result[1],
            metadata={"statistic": adf_result[0], "critical_values": adf_result[4]},
            recommendations=["Series may be non-stationary - consider differencing"] if adf_result[1] >= self.thresholds['alpha_level'] else []
        ))
        
        # KPSS test
        kpss_result = kpss(series)
        results.append(ValidationResult(
            test_name="kpss_stationarity",
            test_type="statistical",
            passed=kpss_result[1] > self.thresholds['alpha_level'],
            score=kpss_result[1],
            p_value=kpss_result[1],
            metadata={"statistic": kpss_result[0], "critical_values": kpss_result[3]},
            recommendations=["Series may be non-stationary"] if kpss_result[1] <= self.thresholds['alpha_level'] else []
        ))
        
        return results
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> List[ValidationResult]:
        """Test autocorrelation in residuals"""
        results = []
        
        # Ljung-Box test
        ljung_box_result = acorr_ljungbox(residuals, lags=10, return_df=True)
        avg_p_value = ljung_box_result['lb_pvalue'].mean()
        
        results.append(ValidationResult(
            test_name="ljung_box_autocorrelation",
            test_type="statistical",
            passed=avg_p_value > self.thresholds['alpha_level'],
            score=avg_p_value,
            p_value=avg_p_value,
            metadata={"test_results": ljung_box_result.to_dict()},
            recommendations=["Residuals show autocorrelation - model may be inadequate"] if avg_p_value <= self.thresholds['alpha_level'] else []
        ))
        
        return results
    
    def _test_heteroscedasticity(self, residuals: np.ndarray, fitted_values: np.ndarray) -> List[ValidationResult]:
        """Test heteroscedasticity in residuals"""
        results = []
        
        # Breusch-Pagan test (simplified)
        # Regress squared residuals on fitted values
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(fitted_values, residuals**2)
            
            results.append(ValidationResult(
                test_name="breusch_pagan_heteroscedasticity",
                test_type="statistical",
                passed=p_value > self.thresholds['alpha_level'],
                score=1 - p_value,
                p_value=p_value,
                metadata={"slope": slope, "r_value": r_value},
                recommendations=["Heteroscedasticity detected - consider weighted regression"] if p_value <= self.thresholds['alpha_level'] else []
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                test_name="heteroscedasticity_error",
                test_type="statistical",
                passed=False,
                score=0.0,
                metadata={"error": str(e)}
            ))
        
        return results
    
    def _test_model_adequacy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           residuals: np.ndarray) -> List[ValidationResult]:
        """Test overall model adequacy"""
        results = []
        
        # R-squared test
        r2 = r2_score(y_true, y_pred)
        results.append(ValidationResult(
            test_name="r_squared_adequacy",
            test_type="statistical",
            passed=r2 > self.thresholds['min_r2'],
            score=r2,
            metadata={"r2": r2, "threshold": self.thresholds['min_r2']},
            recommendations=["Model has low explanatory power"] if r2 <= self.thresholds['min_r2'] else []
        ))
        
        # Mean Squared Error test
        mse = mean_squared_error(y_true, y_pred)
        results.append(ValidationResult(
            test_name="mse_adequacy",
            test_type="statistical",
            passed=mse < self.thresholds['max_mse'],
            score=1 / (1 + mse),
            metadata={"mse": mse, "threshold": self.thresholds['max_mse']},
            recommendations=["Model has high prediction error"] if mse >= self.thresholds['max_mse'] else []
        ))
        
        # Residual randomness test
        residual_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        results.append(ValidationResult(
            test_name="residual_randomness",
            test_type="statistical",
            passed=abs(residual_autocorr) < 0.1,
            score=1 - abs(residual_autocorr),
            metadata={"autocorrelation": residual_autocorr},
            recommendations=["Residuals show systematic patterns"] if abs(residual_autocorr) >= 0.1 else []
        ))
        
        return results
    
    def _run_cross_validation_tests(self, model: Any, X: np.ndarray, y: np.ndarray, 
                                   model_type: str) -> List[ValidationResult]:
        """Run cross-validation tests"""
        results = []
        
        try:
            # Time Series Cross-Validation
            tscv = TimeSeriesSplit(n_splits=self.config['cross_validation']['n_splits'])
            
            # Calculate cross-validation scores
            cv_scores = []
            train_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Calculate scores
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)
                
                train_scores.append(train_score)
                cv_scores.append(val_score)
            
            # Cross-validation performance
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            
            results.append(ValidationResult(
                test_name="time_series_cross_validation",
                test_type="cross_validation",
                passed=mean_cv_score > self.thresholds['min_r2'],
                score=mean_cv_score,
                confidence_interval=(mean_cv_score - 2*std_cv_score, mean_cv_score + 2*std_cv_score),
                metadata={
                    "cv_scores": cv_scores,
                    "mean_score": mean_cv_score,
                    "std_score": std_cv_score,
                    "n_splits": self.config['cross_validation']['n_splits']
                },
                recommendations=["Cross-validation performance is poor"] if mean_cv_score <= self.thresholds['min_r2'] else []
            ))
            
            # Overfitting detection
            mean_train_score = np.mean(train_scores)
            overfitting_score = mean_train_score - mean_cv_score
            
            results.append(ValidationResult(
                test_name="overfitting_detection",
                test_type="cross_validation",
                passed=overfitting_score < self.thresholds['max_overfitting'],
                score=1 - overfitting_score,
                metadata={
                    "train_score": mean_train_score,
                    "cv_score": mean_cv_score,
                    "overfitting_score": overfitting_score
                },
                recommendations=["Model may be overfitting"] if overfitting_score >= self.thresholds['max_overfitting'] else []
            ))
            
            # Score stability
            cv_stability = 1 - (std_cv_score / mean_cv_score) if mean_cv_score != 0 else 0
            results.append(ValidationResult(
                test_name="cross_validation_stability",
                test_type="cross_validation",
                passed=cv_stability > self.thresholds['min_stability'],
                score=cv_stability,
                metadata={
                    "stability_score": cv_stability,
                    "score_std": std_cv_score,
                    "score_mean": mean_cv_score
                },
                recommendations=["Model performance is unstable across folds"] if cv_stability <= self.thresholds['min_stability'] else []
            ))
            
        except Exception as e:
            print(f"⚠️  Error in cross-validation tests: {e}")
            results.append(ValidationResult(
                test_name="cross_validation_error",
                test_type="cross_validation",
                passed=False,
                score=0.0,
                metadata={"error": str(e)}
            ))
        
        return results
    
    def _run_robustness_tests(self, model: Any, X: np.ndarray, y: np.ndarray, 
                             model_type: str) -> List[ValidationResult]:
        """Run robustness tests"""
        results = []
        
        try:
            # 1. Noise Sensitivity Test
            results.extend(self._test_noise_sensitivity(model, X, y))
            
            # 2. Parameter Stability Test
            results.extend(self._test_parameter_stability(model, X, y))
            
            # 3. Outlier Sensitivity Test
            results.extend(self._test_outlier_sensitivity(model, X, y))
            
            # 4. Data Subset Stability Test
            results.extend(self._test_data_subset_stability(model, X, y))
            
        except Exception as e:
            print(f"⚠️  Error in robustness tests: {e}")
            results.append(ValidationResult(
                test_name="robustness_tests_error",
                test_type="robustness",
                passed=False,
                score=0.0,
                metadata={"error": str(e)}
            ))
        
        return results
    
    def _test_noise_sensitivity(self, model: Any, X: np.ndarray, y: np.ndarray) -> List[ValidationResult]:
        """Test sensitivity to noise in input data"""
        results = []
        
        try:
            # Original performance
            original_score = model.score(X, y)
            
            # Add noise and test
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            noise_scores = []
            
            for noise_level in noise_levels:
                # Add Gaussian noise
                X_noise = X + np.random.normal(0, noise_level * np.std(X), X.shape)
                
                # Test performance
                try:
                    noise_score = model.score(X_noise, y)
                    noise_scores.append(noise_score)
                except:
                    noise_scores.append(0.0)
            
            # Calculate average degradation
            avg_degradation = np.mean([original_score - score for score in noise_scores])
            
            results.append(ValidationResult(
                test_name="noise_sensitivity",
                test_type="robustness",
                passed=avg_degradation < 0.1,
                score=1 - avg_degradation,
                metadata={
                    "original_score": original_score,
                    "noise_scores": noise_scores,
                    "avg_degradation": avg_degradation,
                    "noise_levels": noise_levels
                },
                recommendations=["Model is sensitive to input noise"] if avg_degradation >= 0.1 else []
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                test_name="noise_sensitivity_error",
                test_type="robustness",
                passed=False,
                score=0.0,
                metadata={"error": str(e)}
            ))
        
        return results
    
    def _test_parameter_stability(self, model: Any, X: np.ndarray, y: np.ndarray) -> List[ValidationResult]:
        """Test parameter stability across different data subsets"""
        results = []
        
        try:
            # Split data into subsets
            n_subsets = 5
            subset_size = len(X) // n_subsets
            
            parameter_sets = []
            scores = []
            
            for i in range(n_subsets):
                start_idx = i * subset_size
                end_idx = (i + 1) * subset_size if i < n_subsets - 1 else len(X)
                
                X_subset = X[start_idx:end_idx]
                y_subset = y[start_idx:end_idx]
                
                # Fit model on subset
                model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                model_copy.fit(X_subset, y_subset)
                
                # Store parameters if available
                if hasattr(model_copy, 'coef_'):
                    parameter_sets.append(model_copy.coef_)
                
                scores.append(model_copy.score(X_subset, y_subset))
            
            # Calculate parameter stability
            if parameter_sets:
                parameter_stability = 1 - np.std(parameter_sets) / (np.mean(np.abs(parameter_sets)) + 1e-8)
            else:
                parameter_stability = 0.5  # Default if no parameters available
            
            # Score stability
            score_stability = 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) != 0 else 0
            
            results.append(ValidationResult(
                test_name="parameter_stability",
                test_type="robustness",
                passed=parameter_stability > self.thresholds['min_stability'],
                score=parameter_stability,
                metadata={
                    "parameter_stability": parameter_stability,
                    "score_stability": score_stability,
                    "subset_scores": scores
                },
                recommendations=["Model parameters are unstable"] if parameter_stability <= self.thresholds['min_stability'] else []
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                test_name="parameter_stability_error",
                test_type="robustness",
                passed=False,
                score=0.0,
                metadata={"error": str(e)}
            ))
        
        return results
    
    def _test_outlier_sensitivity(self, model: Any, X: np.ndarray, y: np.ndarray) -> List[ValidationResult]:
        """Test sensitivity to outliers"""
        results = []
        
        try:
            # Original performance
            original_score = model.score(X, y)
            
            # Add outliers
            n_outliers = max(1, len(X) // 100)  # 1% outliers
            outlier_indices = np.random.choice(len(X), n_outliers, replace=False)
            
            X_outliers = X.copy()
            y_outliers = y.copy()
            
            # Add extreme values
            X_outliers[outlier_indices] = X_outliers[outlier_indices] + 5 * np.std(X, axis=0)
            y_outliers[outlier_indices] = y_outliers[outlier_indices] + 5 * np.std(y)
            
            # Retrain and test
            model_outliers = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            model_outliers.fit(X_outliers, y_outliers)
            
            # Test on clean data
            outlier_score = model_outliers.score(X, y)
            
            degradation = original_score - outlier_score
            
            results.append(ValidationResult(
                test_name="outlier_sensitivity",
                test_type="robustness",
                passed=degradation < 0.1,
                score=1 - degradation,
                metadata={
                    "original_score": original_score,
                    "outlier_score": outlier_score,
                    "degradation": degradation,
                    "n_outliers": n_outliers
                },
                recommendations=["Model is sensitive to outliers"] if degradation >= 0.1 else []
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                test_name="outlier_sensitivity_error",
                test_type="robustness",
                passed=False,
                score=0.0,
                metadata={"error": str(e)}
            ))
        
        return results
    
    def _test_data_subset_stability(self, model: Any, X: np.ndarray, y: np.ndarray) -> List[ValidationResult]:
        """Test stability across different data subsets"""
        results = []
        
        try:
            # Bootstrap sampling
            n_bootstrap = 10
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X), len(X), replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
                
                # Fit model
                model_bootstrap = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                model_bootstrap.fit(X_bootstrap, y_bootstrap)
                
                # Test on original data
                score = model_bootstrap.score(X, y)
                bootstrap_scores.append(score)
            
            # Calculate stability
            mean_score = np.mean(bootstrap_scores)
            std_score = np.std(bootstrap_scores)
            stability = 1 - (std_score / mean_score) if mean_score != 0 else 0
            
            results.append(ValidationResult(
                test_name="data_subset_stability",
                test_type="robustness",
                passed=stability > self.thresholds['min_stability'],
                score=stability,
                metadata={
                    "bootstrap_scores": bootstrap_scores,
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "stability": stability
                },
                recommendations=["Model is unstable across data subsets"] if stability <= self.thresholds['min_stability'] else []
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                test_name="data_subset_stability_error",
                test_type="robustness",
                passed=False,
                score=0.0,
                metadata={"error": str(e)}
            ))
        
        return results
    
    def _run_performance_tests(self, model: Any, X: np.ndarray, y: np.ndarray, 
                              model_type: str) -> List[ValidationResult]:
        """Run performance-specific tests"""
        results = []
        
        try:
            # Basic performance metrics
            y_pred = model.predict(X)
            
            # R² Score
            r2 = r2_score(y, y_pred)
            results.append(ValidationResult(
                test_name="r2_performance",
                test_type="performance",
                passed=r2 > self.thresholds['min_r2'],
                score=r2,
                metadata={"r2": r2, "threshold": self.thresholds['min_r2']},
                recommendations=["Model has low explanatory power"] if r2 <= self.thresholds['min_r2'] else []
            ))
            
            # Mean Absolute Error
            mae = mean_absolute_error(y, y_pred)
            mae_score = 1 / (1 + mae)
            results.append(ValidationResult(
                test_name="mae_performance",
                test_type="performance",
                passed=mae < np.std(y),
                score=mae_score,
                metadata={"mae": mae, "y_std": np.std(y)},
                recommendations=["Model has high mean absolute error"] if mae >= np.std(y) else []
            ))
            
            # Prediction consistency
            pred_consistency = 1 - (np.std(y_pred) / np.std(y)) if np.std(y) != 0 else 0
            results.append(ValidationResult(
                test_name="prediction_consistency",
                test_type="performance",
                passed=pred_consistency > 0.5,
                score=pred_consistency,
                metadata={"pred_std": np.std(y_pred), "y_std": np.std(y)},
                recommendations=["Predictions lack variability"] if pred_consistency <= 0.5 else []
            ))
            
            # Directional accuracy (for financial models)
            if len(y) > 1:
                y_direction = np.sign(np.diff(y))
                pred_direction = np.sign(np.diff(y_pred))
                directional_accuracy = np.mean(y_direction == pred_direction)
                
                results.append(ValidationResult(
                    test_name="directional_accuracy",
                    test_type="performance",
                    passed=directional_accuracy > 0.5,
                    score=directional_accuracy,
                    metadata={"directional_accuracy": directional_accuracy},
                    recommendations=["Poor directional prediction accuracy"] if directional_accuracy <= 0.5 else []
                ))
            
        except Exception as e:
            print(f"⚠️  Error in performance tests: {e}")
            results.append(ValidationResult(
                test_name="performance_tests_error",
                test_type="performance",
                passed=False,
                score=0.0,
                metadata={"error": str(e)}
            ))
        
        return results
    
    def _calculate_overall_assessment(self, all_results: Dict[str, List[ValidationResult]]) -> Tuple[float, str]:
        """Calculate overall assessment score and status"""
        total_score = 0
        total_weight = 0
        
        # Weight different test categories
        weights = {
            'statistical_tests': 0.3,
            'cross_validation_tests': 0.3,
            'robustness_tests': 0.2,
            'performance_tests': 0.2
        }
        
        for category, results in all_results.items():
            if results:
                category_score = np.mean([r.score for r in results])
                weight = weights.get(category, 0.25)
                total_score += category_score * weight
                total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        # Determine status
        if overall_score >= 0.8:
            status = "PASS"
        elif overall_score >= 0.6:
            status = "WARNING"
        else:
            status = "FAIL"
        
        return overall_score, status
    
    def _generate_recommendations(self, all_results: Dict[str, List[ValidationResult]]) -> Tuple[List[str], List[str], str]:
        """Generate recommendations based on test results"""
        critical_issues = []
        improvement_suggestions = []
        
        for category, results in all_results.items():
            for result in results:
                if not result.passed:
                    critical_issues.extend(result.recommendations)
                elif result.score < 0.8:
                    improvement_suggestions.extend(result.recommendations)
        
        # Remove duplicates
        critical_issues = list(set(critical_issues))
        improvement_suggestions = list(set(improvement_suggestions))
        
        # Deployment recommendation
        failed_tests = sum(sum(1 for r in results if not r.passed) for results in all_results.values())
        total_tests = sum(len(results) for results in all_results.values())
        
        if failed_tests == 0:
            deployment_rec = "RECOMMENDED - All tests passed"
        elif failed_tests / total_tests < 0.2:
            deployment_rec = "CAUTIOUS - Some tests failed, monitor closely"
        else:
            deployment_rec = "NOT RECOMMENDED - Too many test failures"
        
        return critical_issues, improvement_suggestions, deployment_rec
    
    def _create_failed_report(self, model_name: str, error_msg: str) -> ModelValidationReport:
        """Create a failed validation report"""
        return ModelValidationReport(
            model_name=model_name,
            validation_date=datetime.now(),
            overall_score=0.0,
            overall_status="FAIL",
            statistical_tests=[],
            cross_validation_tests=[],
            robustness_tests=[],
            performance_tests=[],
            total_tests=0,
            passed_tests=0,
            failed_tests=1,
            warning_tests=0,
            critical_issues=[f"Validation failed: {error_msg}"],
            improvement_suggestions=[],
            deployment_recommendation="NOT RECOMMENDED - Validation failed"
        )
    
    def generate_validation_report(self, model_name: str, save_to_file: bool = True) -> str:
        """Generate comprehensive validation report"""
        if model_name not in self.test_results:
            return f"No validation results found for {model_name}"
        
        report = self.test_results[model_name]
        
        report_text = f"""
🔬 MODEL VALIDATION REPORT
===========================
Model: {report.model_name}
Validation Date: {report.validation_date.strftime('%Y-%m-%d %H:%M:%S')}
Overall Score: {report.overall_score:.3f}
Overall Status: {report.overall_status}

📊 TEST SUMMARY
Total Tests: {report.total_tests}
Passed: {report.passed_tests}
Failed: {report.failed_tests}
Success Rate: {report.passed_tests/report.total_tests*100:.1f}%

🧪 STATISTICAL TESTS
"""
        
        for test in report.statistical_tests:
            status = "✅ PASS" if test.passed else "❌ FAIL"
            report_text += f"  {test.test_name}: {status} (Score: {test.score:.3f})\n"
        
        report_text += f"""
🔄 CROSS-VALIDATION TESTS
"""
        
        for test in report.cross_validation_tests:
            status = "✅ PASS" if test.passed else "❌ FAIL"
            report_text += f"  {test.test_name}: {status} (Score: {test.score:.3f})\n"
        
        report_text += f"""
🛡️  ROBUSTNESS TESTS
"""
        
        for test in report.robustness_tests:
            status = "✅ PASS" if test.passed else "❌ FAIL"
            report_text += f"  {test.test_name}: {status} (Score: {test.score:.3f})\n"
        
        report_text += f"""
📈 PERFORMANCE TESTS
"""
        
        for test in report.performance_tests:
            status = "✅ PASS" if test.passed else "❌ FAIL"
            report_text += f"  {test.test_name}: {status} (Score: {test.score:.3f})\n"
        
        report_text += f"""
🚨 CRITICAL ISSUES
"""
        
        if report.critical_issues:
            for issue in report.critical_issues:
                report_text += f"  • {issue}\n"
        else:
            report_text += "  None identified\n"
        
        report_text += f"""
💡 IMPROVEMENT SUGGESTIONS
"""
        
        if report.improvement_suggestions:
            for suggestion in report.improvement_suggestions:
                report_text += f"  • {suggestion}\n"
        else:
            report_text += "  None identified\n"
        
        report_text += f"""
🚀 DEPLOYMENT RECOMMENDATION
{report.deployment_recommendation}

📋 INTERPRETATION
"""
        
        if report.overall_status == "PASS":
            report_text += "🟢 MODEL READY: All validation tests passed successfully\n"
        elif report.overall_status == "WARNING":
            report_text += "🟡 PROCEED WITH CAUTION: Some tests failed, monitor closely\n"
        else:
            report_text += "🔴 NOT READY: Model failed validation, requires improvement\n"
        
        report_text += f"""
⚠️  VALIDATION NOTES
• This validation is based on historical data
• Model performance may vary in live trading
• Regular revalidation is recommended
• Consider ensemble methods for improved robustness
• Monitor model drift and performance degradation
"""
        
        # Save to file
        if save_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'model_validation_{model_name}_{timestamp}.txt'
            with open(filename, 'w') as f:
                f.write(report_text)
            print(f"📄 Validation report saved to {filename}")
        
        return report_text
    
    def create_validation_dashboard(self, model_name: str) -> Dict[str, Any]:
        """Create interactive validation dashboard"""
        if model_name not in self.test_results:
            return {}
        
        report = self.test_results[model_name]
        
        # Prepare data for visualization
        categories = ['Statistical', 'Cross-Validation', 'Robustness', 'Performance']
        test_counts = [
            len(report.statistical_tests),
            len(report.cross_validation_tests),
            len(report.robustness_tests),
            len(report.performance_tests)
        ]
        
        passed_counts = [
            sum(1 for t in report.statistical_tests if t.passed),
            sum(1 for t in report.cross_validation_tests if t.passed),
            sum(1 for t in report.robustness_tests if t.passed),
            sum(1 for t in report.performance_tests if t.passed)
        ]
        
        dashboard_data = {
            'model_name': model_name,
            'overall_score': report.overall_score,
            'overall_status': report.overall_status,
            'categories': categories,
            'test_counts': test_counts,
            'passed_counts': passed_counts,
            'failed_counts': [total - passed for total, passed in zip(test_counts, passed_counts)],
            'test_details': {
                'statistical': report.statistical_tests,
                'cross_validation': report.cross_validation_tests,
                'robustness': report.robustness_tests,
                'performance': report.performance_tests
            }
        }
        
        return dashboard_data

def main():
    """Demo function for model validation framework"""
    print("🔬 MODEL VALIDATION FRAMEWORK DEMO")
    print("=" * 60)
    
    # Create synthetic model and data for testing
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(1000) * 0.1
    
    # Create models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Initialize validation framework
    validator = ModelValidationFramework()
    
    # Validate each model
    for model_name, model in models.items():
        print(f"\n🧪 Validating {model_name}...")
        
        # Fit model
        model.fit(X, y)
        
        # Run validation
        validation_report = validator.validate_model(model, X, y, model_name)
        
        print(f"✅ Validation completed for {model_name}")
        print(f"   Overall Score: {validation_report.overall_score:.3f}")
        print(f"   Status: {validation_report.overall_status}")
        print(f"   Tests Passed: {validation_report.passed_tests}/{validation_report.total_tests}")
        
        # Generate report
        report_text = validator.generate_validation_report(model_name, save_to_file=False)
        print(f"\n📋 Validation Report Preview:")
        print(report_text[:500] + "..." if len(report_text) > 500 else report_text)
        
        # Create dashboard
        dashboard = validator.create_validation_dashboard(model_name)
        if dashboard:
            print(f"📊 Dashboard created with {len(dashboard)} metrics")
    
    print("\n" + "=" * 60)
    print("🎯 Model Validation Framework Demo Complete!")

if __name__ == "__main__":
    main() 
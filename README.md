# FitFitFitter

Interactive Distribution Fitting and Statistical Analysis Tool

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Streamlit-based web application for performing statistical analysis and fitting probability distributions to 1D numerical datasets. FitFitFitter provides comprehensive tools for distribution fitting, goodness-of-fit testing, and visual diagnostics.

## 🔗 Repository

**GitHub**: [https://github.com/lhjelm-dk/FitFitFitter](https://github.com/lhjelm-dk/FitFitFitter)

## 🎯 Features

- **Multiple Data Input Methods**: Upload CSV/TXT/XLSX/JSON/Parquet files, paste comma-separated or tabular values, or use the test dataset
- **Comprehensive Descriptive Statistics**: Mean, mode, min, max, P10, P50, P90, standard deviation, skewness, and kurtosis
- **Automatic Distribution Fitting**: Fits 24 different distributions and ranks them by goodness-of-fit
- **Advanced Goodness-of-Fit Metrics**: KS statistic, P-value, AIC, Chi-square, and Anderson-Darling tests
- **Interactive Manual Fitting**: Interactive curve fitting with sliders and manual parameter entry
- **Visual Diagnostics**: Q-Q plots, P-P plots, Histogram+PDF overlays, and CDF Difference plots
- **Excel Export (BETA)**: Export comprehensive reports with formatted data, statistics, and charts
- **Distribution Formulas**: LaTeX and Excel-compatible formulas for all fitted distributions

## 📂 Project Structure

```
FitFitFitter/
├── app.py                      # Main Streamlit application
├── data/
│   └── default_data.csv        # Test dataset
├── utils/
│   ├── __init__.py
│   └── fitting.py              # Distribution fitting and ranking logic
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lhjelm-dk/FitFitFitter.git
   cd FitFitFitter
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

The app will open in your default web browser at `http://localhost:8501`.

### Alternative: Using a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Usage

### Data Input

1. **Upload File**: Upload CSV, TXT, XLSX, JSON, or Parquet files
   - Preview shows first 10 rows
   - Smart column detection for numeric columns
   - Optional data cleaning: remove non-numeric values, zeros, or clip outliers
   - Session state caching with "Reload Last Data" option
2. **Paste Values**: Paste comma-separated values or tabular data (e.g., column from Excel)
3. **Use Test Data**: Load the included test dataset for exploration

**Requirements**:
- At least 10 valid numeric entries
- Values must be non-negative (filtered automatically)

### Features

#### 1. Descriptive Statistics
View comprehensive statistics including:
- Count, Mean, Mode, Minimum, Maximum
- Percentiles: P10, P50 (median), P90
- Standard Deviation, Skewness, Kurtosis
- Dynamic recommendations based on data shape characteristics

#### 2. Automatic Distribution Fitting
Click "Fit Distributions" to automatically fit your data to 24 distributions including:
- Normal, Lognormal, Exponential, Weibull, Gamma, Beta
- Triangular, Logistic, GEV, Pareto, Uniform, Inverse Gaussian, Burr
- Rayleigh, Nakagami, Laplace, Gumbel, Log-Logistic, Cauchy
- Half-Normal, Maxwell, Generalized Pareto, Johnson SU, Johnson SB

Results are ranked by Kolmogorov-Smirnov (KS) statistic (lower is better).

Each distribution shows:
- **Goodness-of-Fit Metrics**: KS statistic, P-value, AIC, Chi-square, Anderson-Darling
- **Statistical Parameters**: Mean, Mode, P10, P50, P90, Min, Max for fitted distribution
- **Visual Diagnostics**: Q-Q plots, P-P plots, Histogram+PDF overlays, CDF Difference plots
- **Distribution Formulas**: LaTeX and Excel-compatible PDF formulas

#### 3. Interactive Cumulative Plots
Visual comparison of empirical data with fitted distributions:
- Cumulative distribution functions (CDF) with hover tooltips showing values and percentiles
- Histogram and PDF density overlays
- P10, P50, P90, and Mean markers on curves
- Log scale option for x-axis
- Customizable x-axis range and number of bins

#### 4. Interactive Manual Fit
Interactively adjust distribution parameters:
1. Select a distribution type from the dropdown
2. Adjust parameters using sliders or enter values manually (allows values outside slider range)
3. See the curve update in real-time on the cumulative plot
4. View comprehensive fit statistics including KS, P-value, AIC, Anderson-Darling
5. See calculated Mean, Mode, P10, P50, P90, Min, Max for the fitted distribution

#### 5. Excel Export (BETA)
Download comprehensive Excel reports including:
- **Report Sheet**: Summary with descriptive statistics and top fitted distributions
- **Data Sheet**: Raw data, sorted data, and empirical CDFs
- **Fitted Curves Sheet**: X values, CDFs, and PDFs for all distributions (ready for charting)
- **Statistics Sheet**: All fit metrics and parameters
- **Quantiles Sheet**: P10, P25, P50, P75, P90, Mean, Mode, Min, Max for each distribution
- **Charts Sheet**: Interactive charts (may require manual creation in some Excel versions)

**Note**: Excel export is in BETA. Charts may not display correctly in all Excel versions. All data is formatted and ready for manual chart creation.

## 🧮 Mathematical Background

### Cumulative Distribution Functions (CDF)
- **Empirical CDF**: `F_emp(x) = (number of values ≤ x) / n`
- **Theoretical CDF**: `F_theo(x) = dist.cdf(x, *params)`
- Plots show cumulative probability vs. value

### Goodness-of-Fit Tests

1. **Kolmogorov-Smirnov (KS) Test**
   - Measures maximum difference between empirical and theoretical CDFs
   - Lower KS statistic indicates better fit
   - P-value > 0.05 suggests good fit

2. **Akaike Information Criterion (AIC)**
   - `AIC = 2k - 2ln(L)` where k = number of parameters, L = likelihood
   - Lower AIC indicates better model (penalizes complexity)

3. **Chi-Square Test**
   - Compares observed vs. expected frequencies in bins
   - Higher P-value indicates better fit

4. **Anderson-Darling Test**
   - More sensitive to tail differences than KS test
   - Lower statistic indicates better fit

### Visual Diagnostics

- **Q-Q Plot**: Quantiles of data vs. quantiles of fitted distribution (good fit = diagonal line)
- **P-P Plot**: Empirical probabilities vs. theoretical probabilities (good fit = diagonal line)
- **CDF Difference Plot**: Difference between theoretical and empirical CDFs (good fit = flat line at y=0)
- **Histogram + PDF Overlay**: Visual comparison of data frequency vs. fitted probability density

## 🧪 Testing

The test dataset includes 36 values with various characteristics. Test with:
- Highly skewed datasets
- Normal/bell-shaped distributions
- Bimodal datasets (may require more advanced distributions)

## 📋 Dependencies

- Python 3.10+
- Streamlit >= 1.28.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- Plotly >= 5.0.0
- Seaborn >= 0.12.0
- openpyxl >= 3.0.0 (for Excel export)
- pyarrow >= 10.0.0 (for Parquet file support)

## 🔧 Technical Details

### Distribution Parameters
Different distributions use different parameter formats:
- **Normal**: (loc, scale) = (μ, σ)
- **Lognormal**: (shape, loc, scale)
- **Exponential**: (loc, scale)
- **Weibull**: (shape, loc, scale)
- **Gamma**: (shape, loc, scale)
- **Beta**: (shape_a, shape_b, loc, scale)

### File Formats
- **CSV/TXT**: Comma-separated or newline-separated values
- **XLSX**: Excel files with automatic numeric column detection
- **JSON**: JSON files with array or object structures
- **Parquet**: Parquet files with numeric columns

All formats support column selection and data cleaning options.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Inspired by the interactive distribution fitting concept from Ariane Logix.

## 📧 Contact

For questions or issues, please open an issue on [GitHub](https://github.com/lhjelm-dk/FitFitFitter/issues).


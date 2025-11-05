"""
Distribution fitting and ranking utilities for fitfitfit
"""

import numpy as np
from scipy import stats
import pandas as pd


# List of distributions to fit
DISTRIBUTIONS = [
    {'name': 'Normal', 'func': stats.norm},
    {'name': 'Lognormal', 'func': stats.lognorm},
    {'name': 'Exponential', 'func': stats.expon},
    {'name': 'Weibull (minimum)', 'func': stats.weibull_min},
    {'name': 'Gamma', 'func': stats.gamma},
    {'name': 'Beta', 'func': stats.beta},
    {'name': 'Triangular', 'func': stats.triang},
    {'name': 'Logistic', 'func': stats.logistic},
    {'name': 'Generalized Extreme Value (GEV)', 'func': stats.genextreme},
    {'name': 'Pareto', 'func': stats.pareto},
    {'name': 'Uniform', 'func': stats.uniform},
    {'name': 'Inverse Gaussian', 'func': stats.invgauss},
    {'name': 'Burr', 'func': stats.burr},
    {'name': 'Rayleigh', 'func': stats.rayleigh},
    {'name': 'Nakagami', 'func': stats.nakagami},
    {'name': 'Laplace', 'func': stats.laplace},
    {'name': 'Gumbel (Right)', 'func': stats.gumbel_r},
    {'name': 'Log-Logistic', 'func': stats.fisk},
    {'name': 'Cauchy', 'func': stats.cauchy},
    {'name': 'Half-Normal', 'func': stats.halfnorm},
    {'name': 'Maxwell', 'func': stats.maxwell},
    {'name': 'Generalized Pareto', 'func': stats.genpareto},
    {'name': 'Johnson SU', 'func': stats.johnsonsu},
    {'name': 'Johnson SB', 'func': stats.johnsonsb},
]


def fit_distribution(data, dist_func):
    """
    Fit a distribution to data and return parameters and KS statistic.
    
    Parameters:
    -----------
    data : array-like
        Input data to fit
    dist_func : scipy.stats distribution
        Distribution function to fit
        
    Returns:
    --------
    params : tuple
        Fitted parameters
    ks_stat : float
        Kolmogorov-Smirnov statistic
    p_value : float
        KS test p-value
    """
    try:
        # Fit distribution
        params = dist_func.fit(data)
        
        # Perform KS test using the distribution's cdf method
        # kstest accepts a callable that takes x as first argument
        ks_stat, p_value = stats.kstest(data, lambda x: dist_func.cdf(x, *params))
        
        return params, ks_stat, p_value
    except Exception as e:
        # Return large KS stat if fitting fails
        return None, np.inf, 0.0


def rank_distributions(data):
    """
    Fit all distributions and rank by KS statistic.
    
    Parameters:
    -----------
    data : array-like
        Input data to fit
        
    Returns:
    --------
    results : list of dicts
        Each dict contains: name, params, ks_stat, p_value, aic, ad_stat, chi2_stat, chi2_pvalue
    """
    results = []
    
    # Map distribution names to Anderson-Darling test names (only some are supported)
    ad_dist_map = {
        'Normal': 'norm',
        'Exponential': 'expon',
        'Logistic': 'logistic',
        'Lognormal': None,  # Not directly supported
        'Weibull (minimum)': None,
        'Gamma': None,
        'Beta': None,
        'Triangular': None,
        'Generalized Extreme Value (GEV)': None,
        'Pareto': None,
        'Uniform': None,
        'Inverse Gaussian': None,
        'Burr': None,
    }
    
    for dist_info in DISTRIBUTIONS:
        params, ks_stat, p_value = fit_distribution(data, dist_info['func'])
        
        if params is not None:
            dist_func = dist_info['func']
            dist_name = dist_info['name']
            
            # Compute log-likelihood for AIC
            try:
                log_likelihood = np.sum(dist_func.logpdf(data, *params))
                
                # Compute number of parameters
                k = len(params)
                
                # AIC calculation: AIC = 2k - 2ln(L)
                aic = 2 * k - 2 * log_likelihood
            except Exception:
                aic = np.nan
            
            # Anderson-Darling test (only works for specific distributions)
            ad_stat = np.nan
            try:
                ad_dist_name = ad_dist_map.get(dist_name)
                if ad_dist_name is not None:
                    ad_result = stats.anderson(data, dist=ad_dist_name)
                    ad_stat = ad_result.statistic
            except Exception:
                ad_stat = np.nan
            
            # Chi-square goodness-of-fit test
            chi2_stat = np.nan
            chi2_pvalue = np.nan
            try:
                # Use adaptive binning to ensure sufficient expected frequencies
                n = len(data)
                n_bins = max(int(np.sqrt(n)), 10)  # Adaptive bin count
                n_bins = min(n_bins, int(n / 5))  # Ensure at least 5 observations per bin on average
                
                # Create histogram
                hist_counts, bin_edges = np.histogram(data, bins=n_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Calculate expected frequencies
                bin_probs = np.diff(dist_func.cdf(bin_edges, *params))
                expected_freq = bin_probs * n
                
                # Filter bins with sufficient expected frequency (>= 5)
                valid_mask = expected_freq >= 5
                if np.sum(valid_mask) >= 3:  # Need at least 3 valid bins
                    obs_freq = hist_counts[valid_mask]
                    exp_freq = expected_freq[valid_mask]
                    
                    # Calculate Chi-square statistic
                    chi2_stat = np.sum((obs_freq - exp_freq) ** 2 / exp_freq)
                    
                    # Degrees of freedom: number of bins - 1 - number of fitted parameters
                    dof = np.sum(valid_mask) - 1 - len(params)
                    if dof > 0:
                        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, dof)
                    else:
                        chi2_pvalue = np.nan
            except Exception:
                chi2_stat = np.nan
                chi2_pvalue = np.nan
            
            results.append({
                'name': dist_name,
                'params': params,
                'ks_stat': ks_stat,
                'p_value': p_value,
                'aic': aic,
                'ad_stat': ad_stat,
                'chi2_stat': chi2_stat,
                'chi2_pvalue': chi2_pvalue,
                'dist_func': dist_func
            })
    
    # Sort by KS statistic (lower is better)
    results.sort(key=lambda x: x['ks_stat'])
    
    return results


def format_params(dist_name, params):
    """
    Format distribution parameters for display.
    
    Parameters:
    -----------
    dist_name : str
        Name of the distribution
    params : tuple
        Fitted parameters (shape, loc, scale) or similar
        
    Returns:
    --------
    formatted : str
        Formatted parameter string
    """
    if dist_name == 'Normal':
        # (loc, scale)
        return f"μ={params[0]:.4f}, σ={params[1]:.4f}"
    elif dist_name == 'Lognormal':
        # (s, loc, scale) where s is shape
        return f"s={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif dist_name == 'Exponential':
        # (loc, scale)
        return f"loc={params[0]:.4f}, scale={params[1]:.4f}"
    elif dist_name == 'Weibull (minimum)':
        # (c, loc, scale) where c is shape
        return f"c={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif dist_name == 'Gamma':
        # (a, loc, scale) where a is shape
        return f"a={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif dist_name == 'Beta':
        # (a, b, loc, scale) where a and b are shape parameters
        return f"a={params[0]:.4f}, b={params[1]:.4f}, loc={params[2]:.4f}, scale={params[3]:.4f}"
    elif dist_name == 'Triangular':
        # (c, loc, scale) where c is shape parameter
        return f"c={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif dist_name == 'Logistic':
        # (loc, scale)
        return f"loc={params[0]:.4f}, scale={params[1]:.4f}"
    elif dist_name == 'Generalized Extreme Value (GEV)':
        # (c, loc, scale) where c is shape parameter
        return f"c={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif dist_name == 'Pareto':
        # (b, loc, scale) where b is shape parameter
        return f"b={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif dist_name == 'Uniform':
        # (loc, scale) where scale is the width
        return f"loc={params[0]:.4f}, scale={params[1]:.4f}"
    elif dist_name == 'Inverse Gaussian':
        # (mu, loc, scale) where mu is shape parameter
        return f"μ={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif dist_name == 'Burr':
        # (c, d, loc, scale) where c and d are shape parameters
        return f"c={params[0]:.4f}, d={params[1]:.4f}, loc={params[2]:.4f}, scale={params[3]:.4f}"
    elif dist_name == 'Rayleigh':
        # (loc, scale) where scale is the mode parameter
        return f"loc={params[0]:.4f}, scale={params[1]:.4f}"
    elif dist_name == 'Nakagami':
        # (nu, loc, scale) where nu is shape parameter
        return f"ν={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif dist_name == 'Laplace':
        # (loc, scale) where scale is the diversity parameter
        return f"loc={params[0]:.4f}, scale={params[1]:.4f}"
    elif dist_name == 'Gumbel (Right)':
        # (loc, scale)
        return f"loc={params[0]:.4f}, scale={params[1]:.4f}"
    elif dist_name == 'Log-Logistic':
        # (c, loc, scale) where c is shape parameter
        return f"c={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif dist_name == 'Cauchy':
        # (loc, scale)
        return f"loc={params[0]:.4f}, scale={params[1]:.4f}"
    elif dist_name == 'Half-Normal':
        # (loc, scale)
        return f"loc={params[0]:.4f}, scale={params[1]:.4f}"
    elif dist_name == 'Maxwell':
        # (loc, scale)
        return f"loc={params[0]:.4f}, scale={params[1]:.4f}"
    elif dist_name == 'Generalized Pareto':
        # (c, loc, scale) where c is shape parameter
        return f"c={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif dist_name == 'Johnson SU':
        # (a, b, loc, scale) where a and b are shape parameters
        return f"a={params[0]:.4f}, b={params[1]:.4f}, loc={params[2]:.4f}, scale={params[3]:.4f}"
    elif dist_name == 'Johnson SB':
        # (a, b, loc, scale) where a and b are shape parameters
        return f"a={params[0]:.4f}, b={params[1]:.4f}, loc={params[2]:.4f}, scale={params[3]:.4f}"
    else:
        return str(params)


def get_param_names(dist_name):
    """
    Get parameter names for a distribution.
    
    Parameters:
    -----------
    dist_name : str
        Name of the distribution
        
    Returns:
    --------
    param_names : list
        List of parameter names
    """
    if dist_name == 'Normal':
        return ['loc (μ)', 'scale (σ)']
    elif dist_name == 'Lognormal':
        return ['shape (s)', 'loc', 'scale']
    elif dist_name == 'Exponential':
        return ['loc', 'scale (λ)']
    elif dist_name == 'Weibull (minimum)':
        return ['shape (c)', 'loc', 'scale']
    elif dist_name == 'Gamma':
        return ['shape (a)', 'loc', 'scale']
    elif dist_name == 'Beta':
        return ['shape (a)', 'shape (b)', 'loc', 'scale']
    elif dist_name == 'Triangular':
        return ['shape (c)', 'loc', 'scale']
    elif dist_name == 'Logistic':
        return ['loc', 'scale']
    elif dist_name == 'Generalized Extreme Value (GEV)':
        return ['shape (c)', 'loc', 'scale']
    elif dist_name == 'Pareto':
        return ['shape (b)', 'loc', 'scale']
    elif dist_name == 'Uniform':
        return ['loc', 'scale']
    elif dist_name == 'Inverse Gaussian':
        return ['shape (μ)', 'loc', 'scale']
    elif dist_name == 'Burr':
        return ['shape (c)', 'shape (d)', 'loc', 'scale']
    elif dist_name == 'Rayleigh':
        return ['loc', 'scale']
    elif dist_name == 'Nakagami':
        return ['shape (ν)', 'loc', 'scale']
    elif dist_name == 'Laplace':
        return ['loc', 'scale']
    elif dist_name == 'Gumbel (Right)':
        return ['loc', 'scale']
    elif dist_name == 'Log-Logistic':
        return ['shape (c)', 'loc', 'scale']
    elif dist_name == 'Cauchy':
        return ['loc', 'scale']
    elif dist_name == 'Half-Normal':
        return ['loc', 'scale']
    elif dist_name == 'Maxwell':
        return ['loc', 'scale']
    elif dist_name == 'Generalized Pareto':
        return ['shape (c)', 'loc', 'scale']
    elif dist_name == 'Johnson SU':
        return ['shape (a)', 'shape (b)', 'loc', 'scale']
    elif dist_name == 'Johnson SB':
        return ['shape (a)', 'shape (b)', 'loc', 'scale']
    else:
        return []


def get_default_bounds(dist_name, data):
    """
    Get default parameter bounds for sliders based on data.
    
    Parameters:
    -----------
    dist_name : str
        Name of the distribution
    data : array-like
        Input data
        
    Returns:
    --------
    bounds : dict
        Dictionary with parameter names as keys and (min, max) tuples as values
    """
    data_min = np.min(data)
    data_max = np.max(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_range = data_max - data_min
    
    # Calculate more appropriate bounds based on data characteristics
    # For loc parameters, use data range with some padding
    loc_min = max(data_min - data_range * 0.5, data_min - data_std * 3)
    loc_max = min(data_max + data_range * 0.5, data_max + data_std * 3)
    
    # For scale parameters, use data standard deviation or range as appropriate
    # Scale should typically be positive and related to data spread
    scale_min = max(0.001, data_std * 0.01)  # Very small minimum
    scale_max = max(data_std * 10, data_range * 2)  # Generous maximum
    
    bounds = {}
    
    if dist_name == 'Normal':
        bounds['loc (μ)'] = (loc_min, loc_max)
        bounds['scale (σ)'] = (scale_min, min(scale_max, data_std * 5))
    elif dist_name == 'Lognormal':
        bounds['shape (s)'] = (0.1, 5.0)
        bounds['loc'] = (loc_min, min(loc_max, data_max + data_std))
        bounds['scale'] = (max(0.1, data_mean * 0.1), max(data_mean * 2, data_max * 1.5))
    elif dist_name == 'Exponential':
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        # For exponential, scale is related to mean (1/rate)
        bounds['scale (λ)'] = (0.01, max(data_mean * 5, data_max * 2))
    elif dist_name == 'Weibull (minimum)':
        bounds['shape (c)'] = (0.1, 10.0)
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_std * 0.1), max(data_range * 2, data_max * 3))
    elif dist_name == 'Gamma':
        bounds['shape (a)'] = (0.1, 20.0)
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        # For Gamma, scale relates to variance/mean ratio
        if data_mean > 0:
            bounds['scale'] = (max(0.1, data_std * 0.1), max(data_std * 3, data_mean * 2))
        else:
            bounds['scale'] = (0.1, scale_max)
    elif dist_name == 'Beta':
        # Beta distribution needs to be scaled to fit data range
        bounds['shape (a)'] = (0.1, 20.0)
        bounds['shape (b)'] = (0.1, 200.0)  # Wider range for shape (b)
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_range * 0.1), data_range * 5)
    elif dist_name == 'Triangular':
        bounds['shape (c)'] = (0.0, 1.0)  # c is between 0 and 1
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_range * 0.1), data_range * 4)
    elif dist_name == 'Logistic':
        bounds['loc'] = (loc_min, loc_max)
        bounds['scale'] = (scale_min, min(scale_max, data_std * 3))
    elif dist_name == 'Generalized Extreme Value (GEV)':
        bounds['shape (c)'] = (-5.0, 5.0)  # Shape parameter can be negative or positive
        bounds['loc'] = (loc_min, loc_max)
        bounds['scale'] = (scale_min, min(scale_max, data_std * 3))
    elif dist_name == 'Pareto':
        bounds['shape (b)'] = (0.1, 10.0)
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_std * 0.1), max(data_range * 1.5, data_max * 2))
    elif dist_name == 'Uniform':
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_range * 0.1), data_range * 3)
    elif dist_name == 'Inverse Gaussian':
        bounds['shape (μ)'] = (max(0.1, data_mean * 0.1), data_mean * 5)
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (scale_min, min(scale_max, data_std * 3))
    elif dist_name == 'Burr':
        bounds['shape (c)'] = (0.1, 10.0)
        bounds['shape (d)'] = (0.1, 10.0)
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_std * 0.1), max(data_range * 1.5, data_max * 2))
    elif dist_name == 'Rayleigh':
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_std * 0.1), max(data_std * 3, data_mean * 2))
    elif dist_name == 'Nakagami':
        bounds['shape (ν)'] = (0.5, 10.0)  # nu >= 0.5
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_std * 0.1), max(data_std * 3, data_mean * 2))
    elif dist_name == 'Laplace':
        bounds['loc'] = (loc_min, loc_max)
        bounds['scale'] = (scale_min, min(scale_max, data_std * 3))
    elif dist_name == 'Gumbel (Right)':
        bounds['loc'] = (loc_min, loc_max)
        bounds['scale'] = (scale_min, min(scale_max, data_std * 3))
    elif dist_name == 'Log-Logistic':
        bounds['shape (c)'] = (0.1, 10.0)
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_mean * 0.1), max(data_mean * 2, data_max * 1.5))
    elif dist_name == 'Cauchy':
        bounds['loc'] = (loc_min, loc_max)
        bounds['scale'] = (scale_min, min(scale_max, data_std * 3))
    elif dist_name == 'Half-Normal':
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (scale_min, min(scale_max, data_std * 3))
    elif dist_name == 'Maxwell':
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_std * 0.1), max(data_std * 3, data_mean * 2))
    elif dist_name == 'Generalized Pareto':
        bounds['shape (c)'] = (-2.0, 2.0)  # Can be negative or positive
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_std * 0.1), max(data_range * 2, data_std * 5))
    elif dist_name == 'Johnson SU':
        bounds['shape (a)'] = (-5.0, 5.0)  # Can be negative or positive
        bounds['shape (b)'] = (0.1, 5.0)  # Must be positive
        bounds['loc'] = (loc_min, loc_max)
        bounds['scale'] = (scale_min, min(scale_max, data_std * 3))
    elif dist_name == 'Johnson SB':
        bounds['shape (a)'] = (-5.0, 5.0)  # Can be negative or positive
        bounds['shape (b)'] = (0.1, 5.0)  # Must be positive
        bounds['loc'] = (loc_min, min(loc_max, data_min + data_std))
        bounds['scale'] = (max(0.1, data_range * 0.1), data_range * 3)
    
    return bounds


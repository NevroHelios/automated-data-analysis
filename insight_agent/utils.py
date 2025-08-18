import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

def compute_distribution_stats(series: pd.Series) -> Dict[str, float]:
    """
    Compute comprehensive distribution statistics for a numerical series.
    
    Returns:
        Dict with mean, std, min, max, quantiles, skewness, kurtosis, outlier count
    """
    try:
        # Ensure purely numeric float series
        clean_series = pd.to_numeric(series, errors='coerce').dropna().astype(float)
        if len(clean_series) == 0:
            return {}
        
        # Basic stats
        basic_stats = {
            'mean': float(clean_series.mean()),
            'std': float(clean_series.std()),
            'min': float(clean_series.min()),
            'max': float(clean_series.max()),
            'count': int(len(clean_series))
        }
        
        # Quantiles
        quantiles = clean_series.quantile([0.25, 0.5, 0.75, 0.95])
        basic_stats.update({
            'q25': float(quantiles[0.25]),
            'median': float(quantiles[0.5]),
            'q75': float(quantiles[0.75]),
            'q95': float(quantiles[0.95])
        })
        
        # Distribution shape (use pandas/numpy to avoid external deps)
        try:
            # pandas uses unbiased estimators by default
            skew_val = np.asarray(clean_series.skew()).astype(float).item()
            # pandas kurtosis is Fisher's definition (0 for normal)
            kurt_val = np.asarray(clean_series.kurtosis()).astype(float).item()
            # Guard NaNs
            basic_stats['skewness'] = float(0.0 if pd.isna(skew_val) else skew_val)
            basic_stats['kurtosis'] = float(0.0 if pd.isna(kurt_val) else kurt_val)
        except Exception:
            basic_stats['skewness'] = 0.0
            basic_stats['kurtosis'] = 0.0
        
        # Outlier detection using IQR method
        try:
            iqr = quantiles[0.75] - quantiles[0.25]
            lower_bound = quantiles[0.25] - 1.5 * iqr
            upper_bound = quantiles[0.75] + 1.5 * iqr
            outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
            basic_stats['outliers'] = int(len(outliers))
        except Exception:
            basic_stats['outliers'] = 0
        
        return basic_stats
    
    except Exception as e:
        logger.error(f"Error computing distribution stats: {e}")
        return {}

def encode_histogram(series: pd.Series, bins: int = 10) -> Dict[str, int]:
    """
    Convert histogram into text-friendly bin counts.
    
    Args:
        series: Numerical series to bin
        bins: Number of bins
        
    Returns:
        Dict mapping bin ranges to counts
    """
    try:
        clean_series = pd.to_numeric(series, errors='coerce').dropna().astype(float)
        if len(clean_series) == 0:
            return {}
        
        # Create histogram
        counts, bin_edges = np.histogram(clean_series, bins=bins)
        
        # Create human-readable bin labels
        histogram_dict = {}
        for i in range(len(counts)):
            left = bin_edges[i]
            right = bin_edges[i + 1]
            
            # Format bin labels nicely
            if abs(left) > 1000 or abs(right) > 1000:
                # Use k notation for thousands
                left_str = f"{left/1000:.0f}k" if left >= 1000 else f"{left:.0f}"
                right_str = f"{right/1000:.0f}k" if right >= 1000 else f"{right:.0f}"
            else:
                left_str = f"{left:.1f}"
                right_str = f"{right:.1f}"
            
            if i == len(counts) - 1:  # Last bin
                bin_label = f"{left_str}+"
            else:
                bin_label = f"{left_str}-{right_str}"
            
            histogram_dict[bin_label] = int(counts[i])
        
        return histogram_dict
    
    except Exception as e:
        logger.error(f"Error encoding histogram: {e}")
        return {}

def compute_correlation_stats(x_series: pd.Series, y_series: pd.Series) -> Dict[str, Any]:
    """
    Compute correlation and regression statistics between two numerical series.
    
    Returns:
        Dict with correlation coefficients, linear fit parameters, and sample points
    """
    try:
        # Build numeric dataframe and drop NaNs
        df_clean = pd.DataFrame({'x': x_series, 'y': y_series})
        df_clean = df_clean.apply(pd.to_numeric, errors='coerce').dropna()
        if len(df_clean) < 3:
            return {}
        
        x_clean = df_clean['x'].to_numpy(dtype=float)
        y_clean = df_clean['y'].to_numpy(dtype=float)
        
        # Correlation coefficients (numpy/pandas implementations)
        try:
            # Pearson via numpy corrcoef
            pearson_matrix = np.corrcoef(x_clean, y_clean)
            pearson_r = float(pearson_matrix[0, 1]) if pearson_matrix.shape == (2, 2) else 0.0
        except Exception:
            pearson_r = 0.0
        try:
            # Spearman via rank correlation
            x_rank = pd.Series(x_clean).rank(method='average').to_numpy(dtype=float)
            y_rank = pd.Series(y_clean).rank(method='average').to_numpy(dtype=float)
            spearman_matrix = np.corrcoef(x_rank, y_rank)
            spearman_r = float(spearman_matrix[0, 1]) if spearman_matrix.shape == (2, 2) else 0.0
        except Exception:
            spearman_r = 0.0
        
        # Simple linear regression (manual calculation)
        try:
            # Convert to numpy arrays for calculations
            x_arr = np.array(x_clean, dtype=float)
            y_arr = np.array(y_clean, dtype=float)
            
            # Calculate slope and intercept using least squares
            n = len(x_arr)
            sum_x = np.sum(x_arr)
            sum_y = np.sum(y_arr)
            sum_xy = np.sum(x_arr * y_arr)
            sum_x2 = np.sum(x_arr * x_arr)
            
            # Slope and intercept
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # R-squared
            y_pred = slope * x_arr + intercept
            ss_res = np.sum((y_arr - y_pred) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
        except Exception:
            slope, intercept, r2 = 0.0, 0.0, 0.0
        
        # Sample points for visualization context
        sample_points = sample_scatter_points(df_clean, max_points=50)
        
        return {
            'pearson_r': float(round(pearson_r, 3)) if isinstance(pearson_r, (int, float)) else 0.0,
            'spearman_r': float(round(spearman_r, 3)) if isinstance(spearman_r, (int, float)) else 0.0,
            'linear_fit': {
                'slope': round(float(slope), 2),
                'intercept': round(float(intercept), 2),
                'r2': round(float(r2), 3)
            },
            'sample_points': sample_points,
            'data_points': len(df_clean)
        }
    
    except Exception as e:
        logger.error(f"Error computing correlation stats: {e}")
        return {}

def sample_scatter_points(df: pd.DataFrame, max_points: int = 100) -> List[List[float]]:
    """
    Sample representative points from scatter plot data.
    
    Args:
        df: DataFrame with 'x' and 'y' columns
        max_points: Maximum number of points to sample
        
    Returns:
        List of [x, y] coordinate pairs
    """
    try:
        if len(df) <= max_points:
            return df[['x', 'y']].values.tolist()
        
        # Stratified sampling - sample from different regions
        sample_size = max_points
        sampled_df = df.sample(n=sample_size, random_state=42)
        
        return [[float(row['x']), float(row['y'])] for _, row in sampled_df.iterrows()]
    
    except Exception as e:
        logger.error(f"Error sampling scatter points: {e}")
        return []

def compute_category_stats(series: pd.Series, target_series: Optional[pd.Series] = None, top_k: int = 10) -> Dict[str, Any]:
    """
    Compute statistics for categorical data.
    
    Args:
        series: Categorical series
        target_series: Optional numerical target for computing averages
        top_k: Number of top categories to include
        
    Returns:
        Dict with category counts and optional target averages
    """
    try:
        clean_series = series.dropna().astype(str)
        if len(clean_series) == 0:
            return {}
        
        # Value counts
        value_counts = clean_series.value_counts().head(top_k)
        category_counts = {str(k): int(v) for k, v in value_counts.items()}
        
        result = {
            'counts': category_counts,
            'total_categories': int(series.nunique()),
            'total_count': int(len(clean_series))
        }
        
        # If target series provided, compute averages
        if target_series is not None:
            try:
                df_clean = pd.DataFrame({
                    'category': clean_series,
                    'target': target_series
                }).dropna()
                
                category_means = df_clean.groupby('category')['target'].mean().sort_values(ascending=False)
                category_averages = {str(k): round(float(v), 2) for k, v in category_means.head(top_k).items()}
                result['averages'] = category_averages
            except Exception:
                pass
        
        return result
    
    except Exception as e:
        logger.error(f"Error computing category stats: {e}")
        return {}

def encode_plot_data(plot_config: Dict, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Encode plot data based on plot type for LLM analysis.
    """
    try:
        def _is_effectively_numeric(series: pd.Series) -> bool:
            s = pd.to_numeric(series, errors='coerce')
            return s.notna().sum() >= max(3, int(0.05 * len(s)))

        def _coerce_numeric(series: pd.Series) -> pd.Series:
            return pd.to_numeric(series, errors='coerce')

        plot_type = plot_config.get('plot_type', '')
        columns = plot_config.get('columns', [])

        if not columns:
            return {}

        available_cols = [col for col in columns if col in data.columns]
        if not available_cols:
            return {}

        encoded_data: Dict[str, Any] = {
            'plot_type': plot_type,
            'columns': available_cols,
            'data_shape': [int(data.shape[0]), int(data.shape[1])]
        }

        if plot_type in ['histogram', 'box', 'violin']:
            col = available_cols[0]
            series = data[col]
            if _is_effectively_numeric(series):
                num_series = _coerce_numeric(series)
                encoded_data['distribution'] = compute_distribution_stats(num_series)
                encoded_data['histogram_bins'] = encode_histogram(num_series)

        elif plot_type == 'scatter':
            if len(available_cols) >= 2:
                x_col, y_col = available_cols[0], available_cols[1]
                x_s, y_s = data[x_col], data[y_col]
                if _is_effectively_numeric(x_s) and _is_effectively_numeric(y_s):
                    encoded_data['correlation'] = compute_correlation_stats(_coerce_numeric(x_s), _coerce_numeric(y_s))

        elif plot_type in ['bar', 'pie']:
            col = available_cols[0]
            target_col = available_cols[1] if len(available_cols) > 1 else None
            target_series = None
            if target_col is not None and _is_effectively_numeric(data[target_col]):
                target_series = _coerce_numeric(data[target_col])
            encoded_data['categories'] = compute_category_stats(data[col], target_series)

        elif plot_type == 'line':
            if len(available_cols) >= 2:
                x_col, y_col = available_cols[0], available_cols[1]
                y_s = data[y_col]
                if _is_effectively_numeric(y_s):
                    y_num = _coerce_numeric(y_s)
                    encoded_data['trend'] = {
                        'y_stats': compute_distribution_stats(y_num),
                        'data_points': int(len(data)),
                        'y_range': [float(y_num.min()), float(y_num.max())]
                    }
                    x_s = data[x_col]
                    if _is_effectively_numeric(x_s):
                        encoded_data['trend']['correlation'] = compute_correlation_stats(_coerce_numeric(x_s), y_num)

        elif plot_type == 'general':
            numeric_stats: Dict[str, Any] = {}
            categorical_stats: Dict[str, Any] = {}
            numeric_histograms: Dict[str, Dict[str, int]] = {}

            for col in available_cols:
                series = data[col]
                if _is_effectively_numeric(series):
                    num_series = _coerce_numeric(series)
                    stats_obj = compute_distribution_stats(num_series)
                    numeric_histograms[col] = encode_histogram(num_series, bins=10)
                    numeric_stats[col] = stats_obj
                else:
                    categorical_stats[col] = compute_category_stats(series)

            if len(numeric_stats) >= 2:
                num_cols = list(numeric_stats.keys())
                corr_candidates: List[tuple] = []
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        x_s = _coerce_numeric(data[num_cols[i]])
                        y_s = _coerce_numeric(data[num_cols[j]])
                        stats_pair = compute_correlation_stats(x_s, y_s)
                        r = abs(stats_pair.get('pearson_r', 0.0))
                        corr_candidates.append((r, num_cols[i], num_cols[j], stats_pair))
                corr_candidates.sort(key=lambda t: t[0], reverse=True)
                top_corrs: List[Dict[str, Any]] = []
                for r, c1, c2, s in corr_candidates[:3]:
                    top_corrs.append({'columns': [c1, c2], 'correlation': s})
                if top_corrs:
                    encoded_data['top_correlations'] = top_corrs

            if numeric_stats:
                encoded_data['numeric_stats'] = numeric_stats
            if numeric_histograms:
                encoded_data['numeric_histograms'] = numeric_histograms
            if categorical_stats:
                encoded_data['categorical_stats'] = categorical_stats

        return encoded_data
    except Exception as e:
        logger.error(f"Error encoding plot data: {e}")
        return {}

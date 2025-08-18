import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)

def compute_distribution_stats(series: pd.Series) -> Dict[str, float]:
    """
    Compute comprehensive distribution statistics for a numerical series.
    
    Returns:
        Dict with mean, std, min, max, quantiles, skewness, kurtosis, outlier count
    """
    try:
        clean_series = series.dropna()
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
        
        # Distribution shape
        try:
            basic_stats['skewness'] = float(stats.skew(clean_series))
            basic_stats['kurtosis'] = float(stats.kurtosis(clean_series))
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
        clean_series = series.dropna()
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
        # Clean data
        df_clean = pd.DataFrame({'x': x_series, 'y': y_series}).dropna()
        if len(df_clean) < 3:
            return {}
        
        x_clean = df_clean['x'].values
        y_clean = df_clean['y'].values
        
        # Correlation coefficients
        try:
            pearson_r, _ = pearsonr(x_clean, y_clean)
            spearman_r, _ = spearmanr(x_clean, y_clean)
        except Exception:
            pearson_r, spearman_r = 0.0, 0.0
        
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
    
    Args:
        plot_config: Plot configuration with type and columns
        data: Source dataframe
        
    Returns:
        Dict with encoded plot data
    """
    try:
        plot_type = plot_config.get('plot_type', '')
        columns = plot_config.get('columns', [])
        
        if not columns or len(columns) == 0:
            return {}
        
        # Validate columns exist
        available_cols = [col for col in columns if col in data.columns]
        if not available_cols:
            return {}
        
        encoded_data = {
            'plot_type': plot_type,
            'columns': available_cols,
            'data_shape': list(data.shape)
        }
        
        if plot_type in ['histogram', 'box', 'violin']:
            # Distribution analysis
            if len(available_cols) >= 1:
                col = available_cols[0]
                if pd.api.types.is_numeric_dtype(data[col]):
                    encoded_data['distribution'] = compute_distribution_stats(data[col])
                    encoded_data['histogram_bins'] = encode_histogram(data[col])
        
        elif plot_type == 'scatter':
            # Correlation analysis
            if len(available_cols) >= 2:
                x_col, y_col = available_cols[0], available_cols[1]
                if (pd.api.types.is_numeric_dtype(data[x_col]) and 
                    pd.api.types.is_numeric_dtype(data[y_col])):
                    encoded_data['correlation'] = compute_correlation_stats(data[x_col], data[y_col])
        
        elif plot_type in ['bar', 'pie']:
            # Categorical analysis
            if len(available_cols) >= 1:
                col = available_cols[0]
                target_col = available_cols[1] if len(available_cols) > 1 else None
                target_series = data[target_col] if target_col and pd.api.types.is_numeric_dtype(data[target_col]) else None
                encoded_data['categories'] = compute_category_stats(data[col], target_series)
        
        elif plot_type == 'line':
            # Time series or ordered data analysis
            if len(available_cols) >= 2:
                x_col, y_col = available_cols[0], available_cols[1]
                if pd.api.types.is_numeric_dtype(data[y_col]):
                    encoded_data['trend'] = {
                        'y_stats': compute_distribution_stats(data[y_col]),
                        'data_points': len(data),
                        'y_range': [float(data[y_col].min()), float(data[y_col].max())]
                    }
                    
                    # Add correlation if x is also numeric
                    if pd.api.types.is_numeric_dtype(data[x_col]):
                        encoded_data['trend']['correlation'] = compute_correlation_stats(data[x_col], data[y_col])
        
        return encoded_data
    
    except Exception as e:
        logger.error(f"Error encoding plot data: {e}")
        return {}

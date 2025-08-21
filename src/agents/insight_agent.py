import pandas as pd
from typing import Dict, List, Any
import json
from datetime import datetime
import logging
from colorama import Fore

from src.agents.insight_utils import encode_plot_data

logger = logging.getLogger(__name__)


class InsightAgent:
    """
    Advanced insight generation agent that processes data and plots to create
    meaningful analytical insights using statistical summaries and LLM analysis.
    """

    def __init__(self, llm_agent):
        """
        Initialize InsightAgent with an LLM agent for text generation.

        Args:
            llm_agent: LLM agent instance with query() method
        """
        self.llm_agent = llm_agent
        self.insights_memory = []  # Store processed insights

    def analyze_query_result(self, user_question: str, data: pd.DataFrame) -> str:
        """
        Analyze a query result dataset and generate insights.

        Args:
            user_question: The original user question
            data: DataFrame containing query results

        Returns:
            String containing comprehensive analysis insights
        """
        try:
            # Create a general plot config to encode the data
            plot_config = {"plot_type": "general", "columns": list(data.columns)}
            print(data.columns)
            print(Fore.GREEN + f"Plot config created: {plot_config}" + Fore.RESET)
            encoded_data = encode_plot_data(plot_config, data)
            print(
                Fore.LIGHTRED_EX
                + f"Encoded data for analysis: {encoded_data}"
                + Fore.RESET
            )
            # Generate various insights
            insights = []

            # Basic data insights with more detail
            insights.append(
                f"Dataset contains {len(data)} rows and {len(data.columns)} columns."
            )

            # Distribution insights for numeric columns with detailed stats
            if encoded_data.get("numeric_stats"):
                insights.extend(self._analyze_distribution(encoded_data))
                
                # Add detailed numeric summaries
                for col, stats in encoded_data["numeric_stats"].items():
                    mean_val = stats.get("mean", 0)
                    median_val = stats.get("median", 0)
                    outliers = stats.get("outliers", 0)
                    insights.append(
                        f"💰 {col}: mean=${mean_val:,.0f}, median=${median_val:,.0f}, outliers={outliers}"
                    )

            # Category insights for categorical columns with percentages
            if encoded_data.get("categorical_stats"):
                insights.extend(self._analyze_categories(encoded_data))

            # Correlation insights if multiple numeric columns
            numeric_cols = [
                col for col in data.columns 
                if data[col].dtype in ["int64", "float64"]
            ]
            if len(numeric_cols) >= 2 and encoded_data.get("top_correlations"):
                top_corr = encoded_data["top_correlations"][0]  # Get strongest correlation
                corr_data = top_corr["correlation"]
                pearson_r = corr_data.get("pearson_r", 0)
                if abs(pearson_r) > 0.1:  # Only mention if there's some correlation
                    col1, col2 = top_corr["columns"]
                    strength = "strong" if abs(pearson_r) > 0.7 else "moderate" if abs(pearson_r) > 0.4 else "weak"
                    direction = "positive" if pearson_r > 0 else "negative"
                    insights.append(
                        f"🔗 {strength.title()} {direction} correlation between {col1} and {col2} (r={pearson_r:.3f})"
                    )

            # Join all insights
            insight_summary = "\n".join(insights)

            # Store in memory
            self.insights_memory.append(
                {
                    "timestamp": datetime.now(),
                    "question": user_question,
                    "insights": insights,
                    "data_shape": data.shape,
                }
            )

            # Keep only last 10 insights
            if len(self.insights_memory) > 10:
                self.insights_memory = self.insights_memory[-10:]

            return insight_summary

        except Exception as e:
            logger.error(f"Error analyzing query result: {e}")
            return f"Basic analysis: Dataset has {len(data)} rows, {len(data.columns)} columns. Column types: {dict(data.dtypes)}"

    def generate_plot_insights(
        self, plot_configs: List[Dict], data: pd.DataFrame, user_question: str
    ) -> List[List[str]]:
        """
        Generate insights from plot configurations and data.

        Args:
            plot_configs: List of plot configuration dictionaries
            data: Source dataframe
            user_question: Original user question

        Returns:
            List of insight lists - one list of insights per plot
        """
        try:
            all_plot_insights = []
            all_insights_combined = []

            for plot_config in plot_configs:
                # Encode plot data with advanced statistics
                encoded_data = encode_plot_data(plot_config, data)

                if not encoded_data:
                    all_plot_insights.append([])
                    continue

                # Generate insights for this specific plot
                plot_insights = self._analyze_encoded_plot(encoded_data, user_question)
                all_plot_insights.append(plot_insights)
                all_insights_combined.extend(plot_insights)

            # Store in memory
            insight_entry = {
                "timestamp": datetime.now(),
                "question": user_question,
                "insights": all_insights_combined,
                "plots_analyzed": len(plot_configs),
            }
            self.insights_memory.append(insight_entry)

            return all_plot_insights

        except Exception as e:
            logger.error(f"Error generating plot insights: {e}")
            return []

    def _analyze_encoded_plot(
        self, encoded_data: Dict, user_question: str
    ) -> List[str]:
        """
        Analyze encoded plot data and generate specific insights.

        Args:
            encoded_data: Encoded plot statistics
            user_question: Original user question

        Returns:
            List of insight strings
        """
        try:
            plot_type = encoded_data.get("plot_type", "")
            insights = []

            if plot_type in ["histogram", "box", "violin"]:
                # Check for distribution analysis
                if "distribution" in encoded_data:
                    insights.extend(self._analyze_distribution(encoded_data))
                
                # For box plots, also check for categorical analysis
                if plot_type == "box" and "categories" in encoded_data:
                    insights.extend(self._analyze_categories(encoded_data))

            elif plot_type == "scatter" and "correlation" in encoded_data:
                insights.extend(self._analyze_correlation(encoded_data))

            elif plot_type in ["bar", "pie"] and "categories" in encoded_data:
                insights.extend(self._analyze_categories(encoded_data))

            elif plot_type == "line" and "trend" in encoded_data:
                insights.extend(self._analyze_trend(encoded_data))
            
            # Handle general analysis for any plot type
            if not insights:
                # Check for numeric stats
                if "numeric_stats" in encoded_data:
                    insights.extend(self._analyze_distribution(encoded_data))
                
                # Check for categorical stats
                if "categorical_stats" in encoded_data:
                    insights.extend(self._analyze_categories(encoded_data))
                
                # Check for correlation data
                if "correlation" in encoded_data:
                    insights.extend(self._analyze_correlation(encoded_data))

            return insights

        except Exception as e:
            logger.error(f"Error analyzing encoded plot: {e}")
            return []

    def _analyze_distribution(self, encoded_data: Dict) -> List[str]:
        """Analyze distribution data and generate insights."""
        insights = []
        
        # Handle different data structures for distribution analysis
        if "distribution" in encoded_data:
            # Direct distribution data from specific plot types
            dist_data = encoded_data.get("distribution", {})
            column = encoded_data.get("columns", ["Unknown"])[0]
        elif "numeric_stats" in encoded_data:
            # General numeric stats from general plot type
            numeric_stats = encoded_data.get("numeric_stats", {})
            if not numeric_stats:
                return insights
            # Get the first numerical column
            column = list(numeric_stats.keys())[0]
            dist_data = numeric_stats[column]
        else:
            return insights
        try:
            # Skewness insights
            skewness = dist_data.get("skewness", 0)
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                insights.append(
                    f"📊 {column} shows strong {direction}-skewed distribution (skew={skewness:.2f})"
                )
            elif abs(skewness) > 0.5:
                direction = "right" if skewness > 0 else "left"
                insights.append(
                    f"📊 {column} has moderate {direction} skew (skew={skewness:.2f})"
                )
            else:
                insights.append(f"📊 {column} distribution is approximately symmetric.\
                                  skewness: {skewness:.2f}")

            # Outlier insights
            outliers = dist_data.get("outliers", 0)
            total_count = dist_data.get("count", 1)
            if outliers > 0:
                outlier_pct = (outliers / total_count) * 100
                insights.append(
                    f"⚠️ Found {outliers} outliers ({outlier_pct:.1f}% of data)"
                )

            # Spread insights
            q25, q75 = dist_data.get("q25", 0), dist_data.get("q75", 0)
            if q25 and q75:
                iqr = q75 - q25
                insights.append(
                    f"📈 Middle 50% of {column} ranges from {q25:,.0f} to {q75:,.0f} (IQR: {iqr:,.0f})"
                )

            # Kurtosis insights (tail behavior)
            kurtosis = dist_data.get("kurtosis", 0)
            if kurtosis > 3:
                insights.append(
                    f"📊 {column} has heavy tails (kurtosis={kurtosis:.2f})"
                )
            elif kurtosis < -1:
                insights.append(
                    f"📊 {column} has light tails (kurtosis={kurtosis:.2f})"
                )

        except Exception as e:
            logger.error(f"Error analyzing distribution: {e}")

        return insights

    def _analyze_correlation(self, encoded_data: Dict) -> List[str]:
        """Analyze correlation data and generate insights."""
        insights = []
        corr_data = encoded_data.get("correlation", {})
        columns = encoded_data.get("columns", ["X", "Y"])

        try:
            x_col, y_col = columns[0], columns[1] if len(columns) > 1 else columns[0]

            # Correlation strength
            pearson_r = corr_data.get("pearson_r", 0)
            spearman_r = corr_data.get("spearman_r", 0)

            # Interpret correlation strength
            if abs(pearson_r) > 0.8:
                strength = "very strong"
            elif abs(pearson_r) > 0.6:
                strength = "strong"
            elif abs(pearson_r) > 0.4:
                strength = "moderate"
            elif abs(pearson_r) > 0.2:
                strength = "weak"
            else:
                strength = "very weak"

            direction = "positive" if pearson_r > 0 else "negative"
            insights.append(
                f"🔗 {strength.title()} {direction} correlation between {x_col} and {y_col} (r={pearson_r:.3f})"
            )

            # Compare Pearson vs Spearman
            if abs(spearman_r - pearson_r) > 0.1:
                insights.append(
                    f"📊 Non-linear relationship detected (Spearman r={spearman_r:.3f} vs Pearson r={pearson_r:.3f})"
                )

            # Linear fit insights
            linear_fit = corr_data.get("linear_fit", {})
            r2 = linear_fit.get("r2", 0)
            slope = linear_fit.get("slope", 0)

            if r2 > 0.5:
                insights.append(f"📈 Linear model explains {r2 * 100:.1f}% of variance")
                if slope != 0:
                    insights.append(
                        f"📊 For each unit increase in {x_col}, {y_col} changes by {slope:.2f} on average"
                    )

        except Exception as e:
            logger.error(f"Error analyzing correlation: {e}")

        return insights

    def _analyze_categories(self, encoded_data: Dict) -> List[str]:
        """Analyze categorical data and generate insights."""
        insights = []
        
        try:
            # Handle both direct categories and categorical_stats structure
            if "categories" in encoded_data:
                cat_data = encoded_data.get("categories", {})
                column = encoded_data.get("columns", ["Category"])[0]
            elif "categorical_stats" in encoded_data:
                # Handle the new structure where categorical_stats contains multiple columns
                categorical_stats = encoded_data.get("categorical_stats", {})
                if not categorical_stats:
                    return insights
                
                # Get the first categorical column
                column = list(categorical_stats.keys())[0]
                cat_data = categorical_stats[column]
            else:
                return insights

            counts = cat_data.get("counts", {})
            total_categories = cat_data.get("total_categories", 0)
            averages = cat_data.get("averages", {})

            if counts:
                # Most common category
                top_category = max(counts.keys(), key=lambda x: counts[x])
                top_count = counts[top_category]
                total_count = sum(counts.values())
                top_pct = (top_count / total_count) * 100

                insights.append(
                    f"🔝 Most common {column}: '{top_category}' ({top_count} cases, {top_pct:.1f}%)"
                )

                # Category distribution
                if len(counts) > 5:
                    insights.append(
                        f"📊 {column} has {total_categories} unique values, showing high diversity"
                    )
                elif len(counts) <= 3:
                    insights.append(
                        f"📊 {column} has only {total_categories} categories, showing low diversity"
                    )

            if averages:
                # Highest and lowest average categories
                highest_cat = max(averages.keys(), key=lambda x: averages[x])
                lowest_cat = min(averages.keys(), key=lambda x: averages[x])
                highest_val = averages[highest_cat]
                lowest_val = averages[lowest_cat]

                ratio = highest_val / lowest_val if lowest_val > 0 else float("inf")
                insights.append(
                    f"📊 Highest average: '{highest_cat}' ({highest_val:,.0f})"
                )
                insights.append(
                    f"📊 Lowest average: '{lowest_cat}' ({lowest_val:,.0f})"
                )

                if ratio > 3:
                    insights.append(
                        f"⚠️ Large variation between categories (ratio: {ratio:.1f}x)"
                    )

        except Exception as e:
            logger.error(f"Error analyzing categories: {e}")

        return insights

    def _analyze_trend(self, encoded_data: Dict) -> List[str]:
        """Analyze trend/time series data and generate insights."""
        insights = []
        trend_data = encoded_data.get("trend", {})
        columns = encoded_data.get("columns", ["X", "Y"])

        try:
            y_stats = trend_data.get("y_stats", {})
            correlation = trend_data.get("correlation", {})

            # Basic trend insights
            y_col = columns[1] if len(columns) > 1 else columns[0]
            mean_val = y_stats.get("mean", 0)
            std_val = y_stats.get("std", 0)

            if std_val > 0:
                cv = std_val / mean_val if mean_val > 0 else 0
                if cv > 0.5:
                    insights.append(f"📊 {y_col} shows high variability (CV={cv:.2f})")
                elif cv < 0.1:
                    insights.append(f"📊 {y_col} is relatively stable (CV={cv:.2f})")

            # Trend direction if correlation available
            if correlation:
                r_value = correlation.get("pearson_r", 0)
                if abs(r_value) > 0.3:
                    direction = "upward" if r_value > 0 else "downward"
                    insights.append(
                        f"📈 Clear {direction} trend in {y_col} (r={r_value:.3f})"
                    )

        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")

        return insights

    def get_recent_insights(self, limit: int = 5) -> List[Dict]:
        """Get recent insights from memory."""
        return self.insights_memory[-limit:] if self.insights_memory else []

    def generate_summary_with_llm(
        self, encoded_plots: List[Dict], user_question: str, context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate advanced summary using LLM with encoded plot data.

        Args:
            encoded_plots: List of encoded plot data
            user_question: Original user question
            context: Additional context

        Returns:
            Dict with summary and suggestions
        """
        try:
            # Prepare encoded data for LLM
            plot_summaries = []
            for plot_data in encoded_plots:
                plot_type = plot_data.get("plot_type", "unknown")
                summary = f"Plot type: {plot_type}"

                if "distribution" in plot_data:
                    dist = plot_data["distribution"]
                    summary += f" | Distribution: mean={dist.get('mean', 0):.1f}, skew={dist.get('skewness', 0):.2f}, outliers={dist.get('outliers', 0)}"

                if "correlation" in plot_data:
                    corr = plot_data["correlation"]
                    summary += f" | Correlation: r={corr.get('pearson_r', 0):.3f}, R²={corr.get('linear_fit', {}).get('r2', 0):.3f}"

                if "categories" in plot_data:
                    cats = plot_data["categories"]
                    counts = cats.get("counts", {})
                    top_cat = (
                        max(counts.keys(), key=lambda x: counts[x])
                        if counts
                        else "None"
                    )
                    summary += (
                        f" | Top category: {top_cat} ({counts.get(top_cat, 0)} cases)"
                    )

                plot_summaries.append(summary)

            # Combine statistical insights with LLM generation
            statistical_insights = []
            for plot_data in encoded_plots:
                plot_insights = self._analyze_encoded_plot(plot_data, user_question)
                statistical_insights.extend(plot_insights)

            # Create LLM prompt with encoded data
            prompt = f"""
You are an expert data analyst. Based on the statistical analysis below, provide:
1. A concise 1-2 sentence insight summary
2. 3 specific follow-up analysis suggestions
3. Add Numeric data to support your claim

QUESTION: {user_question}

STATISTICAL ANALYSIS:
{chr(10).join(statistical_insights)}

PLOT SUMMARIES:
{chr(10).join(plot_summaries)}

CONTEXT: {context}

Respond in JSON format:
{{"summary": "your insight", "suggestions": ["suggestion1", "suggestion2", "suggestion3"]}}
"""

            # Get LLM response
            llm_response = self.llm_agent.query(prompt).strip()

            # Parse response
            if llm_response.startswith("```json"):
                llm_response = llm_response[7:-3]
            elif llm_response.startswith("```"):
                llm_response = llm_response[3:-3]

            try:
                parsed_response = json.loads(llm_response)

                # Combine statistical insights with LLM summary
                final_summary = parsed_response.get("summary", "")
                if statistical_insights:
                    # Add top statistical insights as bullet points
                    final_summary += (
                        f" Key findings: {' • '.join(statistical_insights[:3])}"
                    )

                return {
                    "summary": final_summary,
                    "suggestions": parsed_response.get("suggestions", []),
                    "statistical_insights": statistical_insights,
                }

            except json.JSONDecodeError:
                # Fallback to statistical insights only
                summary = (
                    " • ".join(statistical_insights[:2])
                    if statistical_insights
                    else "Analysis complete."
                )
                return {
                    "summary": summary,
                    "suggestions": [
                        "Explore data patterns",
                        "Check for outliers",
                        "Analyze relationships",
                    ],
                    "statistical_insights": statistical_insights,
                }

        except Exception as e:
            logger.error(f"Error generating LLM summary: {e}")
            return {
                "summary": "Analysis completed with statistical insights.",
                "suggestions": [
                    "Continue exploring the data",
                    "Look for patterns",
                    "Check data quality",
                ],
                "statistical_insights": [],
            }

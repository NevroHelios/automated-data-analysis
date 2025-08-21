from insight_agent import InsightAgent


def main():
    data = {
        "plot_type": "general",
        "columns": ["LotConfig"],
        "data_shape": [1460, 1],
        "categorical_stats": {
            "LotConfig": {
                "counts": {
                    "Inside": 1052,
                    "Corner": 263,
                    "CulDSac": 94,
                    "FR2": 47,
                    "FR3": 4,
                },
                "total_categories": 5,
                "total_count": 1460,
            }
        },
    }

    agent = InsightAgent(llm_agent="ollama")

    insights = agent._analyze_distribution(data)
    print("\n".join(insights))


if __name__ == "__main__":
    main()

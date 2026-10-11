"""Tests that the rendered tables carry the numbers in the result files, aggregated as stated."""

from toolshortlist.report import category_mean, pct, reproduction_table, table


def result_with(category_scores: dict) -> dict:
    """A results dict whose ToolRet category splits have the given task-mean nDCG@10."""
    splits = {category: {"group_mean": {"ndcg@10": score}} for category, score in category_scores.items()}
    return {"benchmarks": {"toolret": {"settings": {"query": {"splits": splits}}}}}


def test_pct():
    assert pct(0.2224) == "22.24"
    assert pct(1.0) == "100.00"


def test_table():
    assert table(["a", "b"], [["1", "2"]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_category_mean_is_the_mean_of_the_three_category_scores():
    result = result_with({"web": 0.10, "code": 0.20, "customized": 0.60})
    per_category, average = category_mean(result, "query")
    assert per_category == [0.10, 0.20, 0.60]
    assert abs(average - 0.30) < 1e-12


def test_reproduction_table_reports_the_difference_and_the_tolerance():
    results = {"run-a": result_with({"web": 0.10, "code": 0.20, "customized": 0.60})}
    published = [
        {"label": "Model A", "run": "run-a", "setting": "query", "source": "Table 4",
         "web": 10.0, "code": 20.0, "customized": 59.0, "average": 29.67},
        {"label": "Model B", "run": "missing-run", "setting": "query", "source": "Table 4",
         "web": 1.0, "code": 1.0, "customized": 1.0, "average": 1.0},
    ]
    rendered = reproduction_table(results, published).splitlines()
    assert len(rendered) == 3  # header, separator, one row: the run without a result file is skipped
    cells = [cell.strip() for cell in rendered[2].strip("|").split("|")]
    # Ours is 30.00, published 29.67: a difference of +0.33, inside the 0.5 tolerance.
    assert cells[3:7] == ["29.67", "30.00", "+0.33", "yes"]
    assert cells[8] == "10.00 / 20.00 / 60.00"

    published[0]["average"] = 29.0  # now the difference is +1.00
    cells = [c.strip() for c in reproduction_table(results, published).splitlines()[2].strip("|").split("|")]
    assert cells[5:7] == ["+1.00", "no"]

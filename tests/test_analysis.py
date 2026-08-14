def test_build_demo_dataset():
    from src.analysis import build_demo_dataset
    df = build_demo_dataset()
    assert "feature1" in df.columns and "label" in df.columns

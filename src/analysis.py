"""Minimal ML data analysis demo.

Run as: python -m src.analysis
"""
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO)


def build_demo_dataset():
    # small synthetic dataset
    df = pd.DataFrame({
        "feature1": [1,2,3,4,5,6,7,8],
        "feature2": [5,3,6,9,2,4,7,1],
        "label":      [0,1,0,1,0,1,0,1],
    })
    return df


def train_and_eval(df):
    X = df[["feature1","feature2"]]
    y = df["label"]
    # Lazy-import heavy ML libs to keep imports lightweight
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    print(classification_report(yte, preds))


def main():
    df = build_demo_dataset()
    train_and_eval(df)


if __name__ == "__main__":
    main()

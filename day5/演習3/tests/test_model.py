import os
import sys
import pytest
import pandas as pd
import numpy as np
import pickle
import time
import json
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
BASELINE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../models/titanic_model_baseline.pkl"
)


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"


def test_model_no_performance_regression(train_model):
    """過去のベースラインモデルと比較して性能劣化がないことを確認するテスト"""
    current_model, X_test, y_test = train_model

    # 現在のモデルの性能を評価
    current_predictions = current_model.predict(X_test)
    current_accuracy = accuracy_score(y_test, current_predictions)

    # ベースラインモデルの存在確認
    if not os.path.exists(BASELINE_MODEL_PATH):
        # ベースラインが存在しない場合は現在のモデルをベースラインとしてコピー
        with open(MODEL_PATH, "rb") as src, open(BASELINE_MODEL_PATH, "wb") as dst:
            dst.write(src.read())
        warnings.warn(
            f"ベースラインモデルが存在しないため現在のモデルをベースラインとして保存しました"
        )
        pytest.skip("ベースラインモデルを新規作成したため、比較テストはスキップします")

    # ベースラインモデルを読み込む
    try:
        with open(BASELINE_MODEL_PATH, "rb") as f:
            baseline_model = pickle.load(f)
    except Exception as e:
        pytest.fail(f"ベースラインモデルの読み込みに失敗しました: {str(e)}")

    # ベースラインモデルの性能を評価
    baseline_predictions = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    # 性能比較のログ出力
    print(f"現在のモデル精度: {current_accuracy:.4f}")
    print(f"ベースラインモデル精度: {baseline_accuracy:.4f}")

    # 現在の性能がベースライン以上であることを確認
    assert current_accuracy >= baseline_accuracy, (
        f"モデル性能が低下しています。"
        f"現在の精度: {current_accuracy:.4f}, "
        f"ベースライン精度: {baseline_accuracy:.4f}"
    )

    # オプション: 性能が大幅に向上した場合はベースラインを更新
    if current_accuracy > baseline_accuracy * 1.05:  # 5%以上の向上
        with open(MODEL_PATH, "rb") as src, open(BASELINE_MODEL_PATH, "wb") as dst:
            dst.write(src.read())
        print(
            f"モデル性能が大幅に向上したため、ベースラインを更新しました: {current_accuracy:.4f}"
        )


def test_update_baseline_model():
    """ベースラインモデルを手動で更新するためのテスト (--update-baseline フラグ付きで実行)"""
    # コマンドラインフラグをチェック
    if not any("--update-baseline" in arg for arg in sys.argv):
        pytest.skip("このテストは --update-baseline フラグがある場合のみ実行されます")

    if not os.path.exists(MODEL_PATH):
        pytest.fail("更新するモデルが存在しません")

    # 現在のモデルをベースラインとして保存
    with open(MODEL_PATH, "rb") as src, open(BASELINE_MODEL_PATH, "wb") as dst:
        dst.write(src.read())

    print(f"ベースラインモデルを更新しました")
    assert os.path.exists(BASELINE_MODEL_PATH), "ベースラインモデルの保存に失敗しました"

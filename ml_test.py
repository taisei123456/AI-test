from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. データの準備（scikit-learnに入っているサンプルデータを使います）
iris = datasets.load_iris()
X = iris.data  # 入力データ（花びらの長さなど 4項目）
y = iris.target # 正解ラベル（0: Setosa, 1: Versicolor, 2: Virginica）

# 2. データを「学習用」と「テスト用」に分ける
# 全部のデータを学習に使ってしまうと、正しく勉強できたかテストできないため分割します
# test_size=0.2 は「20%をテスト用に残す」という意味です
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. モデル（AIの脳みそ）を選んで学習させる
# ここでは「サポートベクターマシン (SVC)」というアルゴリズムを使います
model = DecisionTreeClassifier() 
model.fit(X_train, y_train) # .fit() が「学習しろ！」という命令です

# 4. 未知のデータ（テスト用データ）で予測してみる
predictions = model.predict(X_test) # .predict() が「予測しろ！」という命令です

# 5. 答え合わせ（正解率を表示）
score = accuracy_score(y_test, predictions)
print(f"正解率: {score * 100:.2f}%")

# 実際にどんな予測をしたか一部見てみる
print("-" * 30)
print(f"実際の答え: {y_test[:5]}") # テストデータの最初の5個の正解
print(f"AIの予測 : {predictions[:5]}") # AIが予測した最初の5個
import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\mguzu\PycharmProjects\reshisII\.venv\creditтабла.csv")
# не надо нам это
x = df.drop(columns=["score", "pid", "util_dttm", "create_dttm", "name",
                     "semester_cost_amt", "util_flg", "subside_rate",
                     "semester_cnt", "initial_approved_amt", "approve_dtm"])
y = df["score"]
# числа и категории
numfigures = ["approve_flg", "age"]
catfigures = ["utm_source", "short_nm", "gender_cd", "education_level_code", "speciality", "initial_term"]
# преобразуем числа
x_num = x[numfigures].copy()
# стандартизация
x_num_mean = x_num.mean()
x_num_std = x_num.std()
x_num_scaled = (x_num - x_num_mean) / x_num_std
# обработка категорий
x_cat = pd.get_dummies(x[catfigures], drop_first=True)
# обьединим
X = np.concatenate([x_num_scaled.values, x_cat.values], axis=1)
np.random.seed(42)
train_size = int(0.99 * len(X))
indices = np.random.permutation(len(X))
x_train, x_test = X[indices[:train_size]], X[indices[train_size:]]
y_train, y_test = y.iloc[indices[:train_size]].to_numpy(), y.iloc[indices[train_size:]].to_numpy()
# свободный член
X_train_b = np.c_[np.ones((x_train.shape[0], 1)), x_train]  # добавляем x0 = 1
X_test_b = np.c_[np.ones((x_test.shape[0], 1)), x_test]
# все в числа
X_train_b = X_train_b.astype(np.float64)
y_train = y_train.astype(np.float64)
X_test_b = X_test_b.astype(np.float64)
# ищем лучшие кэфы
theta_best = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
y_pred = X_test_b.dot(theta_best)
# mae
mae = np.mean(np.abs(y_test - y_pred))
print("mae =", mae)



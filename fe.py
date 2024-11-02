import pandas as pd
import sklearn
#срез сначала - df.head, с конца .tail, тоже что у numpy о значениях.
#получить значение можно по квадратным скобкам, .loc
# инфа о таблице - df.info
#создаем столбцы через df[smth] = арг
#isna ищет пустые значения, dropna выкидывает их, fillna заполняет их
#sort_values("Знач",ascending=t/f) сортировка, groupby([Знач],[знач]) - сортировка .agg(добавление агрегатора)
#pivot_table(аргументы (индекс,колонки,значения,агрегаторы)
import numpy as np #numpy массив: атрибуты shape (размер), dtype, ndim(колво измерений массива), tolist(array to list)
#можем указать dtype и привести все элементы в один тип
#arange(стартовое значение, конечное значение, шаг) - генерация numpy списка с заданными параметрами
#linspace(ст.значение, кон.значение, колво элементов) - генерация списка по колву элементов
#arr.mean - срзнач, arr.std - ст.отклонение, np.median.arr - медиана
#reshape - изменение вида матрицы, resize - обрезка и дальнейшее изменение
#axis 0 - строка, axis = 1 - столбец
# sin, cos, log - все работает в numpy
df = pd.read_csv(r"C:\Users\mguzu\PycharmProjects\reshisII\.venv\creditтабла.csv")
#result = (1 + df['education_level_code'].map({'SPECIALTY': 0.2, 'BACHELOR': 0.3, 'MASTER': 0.5}) - df['approve_flg']) / 2
#print(result - df['score']).abs().mean()
#Задача регрессии? Классифификации вряд ли. Точно не кластеризация. Не выявление аномалии.
from sklearn.model_selection import train_test_split
x = df.drop(columns=["score","pid","util_dttm","create_dttm","name"]) #даты и айдишники. нахуй нам это не надо
y = df["score"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
#model = MLPRegressor(hidden_layer_sizes=(1000,),max_iter=100, random_state=42)
#model = LinearRegression()
#model = RandomForestRegressor(n_estimators=300, random_state=49)
#model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.5, random_state=52)
numfigures = ["util_flg","approve_flg","semester_cnt","initial_approved_amt","initial_term","marketing_flag","age","subside_rate"]
catfigures = ["utm_source","speciality","short_nm","gender_cd","education_level_code"]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numfigures),
        ("cat", OneHotEncoder(), catfigures)
    ])
pipeline = Pipeline(steps=[("preprocessor", preprocessor) ,("model", model)])
pipeline.fit(x_train,y_train)
ypred = pipeline.predict(x_test)
mae = mean_absolute_error(y_test, ypred)
print("mae =", mae)
print(x_train.columns)
#scaler = StandardScaler().fit(x_train)
#standartx = scaler.transform(x_train) #но у нас же не все данные - числа!
#ниче не понимаю. все дает по 0.053 и не больше. завтра...
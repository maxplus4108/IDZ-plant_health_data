import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import balanced_accuracy_score
import joblib

# ——————————————————————————————————————————————————————————————
# Настройка внешнего вида приложения
# ——————————————————————————————————————————————————————————————
st.set_page_config(
    page_title="Plant Health Classifier w/ GridSearch",
    layout="wide"
)

# Словарь для перевода английских меток на русский
label_map = {
    'Healthy': 'Здоровое',
    'High Stress': 'Сильный стресс',
    'Moderate Stress': 'Умеренный стресс'
}

# ——————————————————————————————————————————————————————————————
# Функция: строит четыре конвейера (pipeline) для разных алгоритмов
# ——————————————————————————————————————————————————————————————
def build_models(preprocessor):
    """
    Возвращает словарь вида:
      'Model Name': (pipeline, param_grid)
    где pipeline сначала применяет preprocessor, а затем
    обучает классификатор.
    """
    return {
        'Logistic Regression': (
            Pipeline([
                ('preprocessor', preprocessor),                         # шаг предобработки
                ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
            ]),
            {'clf__C': [0.1, 1, 10]}                                    # сетка C для LR
        ),
        'KNN': (
            Pipeline([
                ('preprocessor', preprocessor),
                ('clf', KNeighborsClassifier())
            ]),
            {'clf__n_neighbors': [3, 5, 7]}                             # сетка числа соседей
        ),
        'Decision Tree': (
            Pipeline([
                ('preprocessor', preprocessor),
                ('clf', DecisionTreeClassifier(class_weight='balanced'))
            ]),
            {'clf__max_depth': [3, 5, None]}                            # сетка глубины дерева
        ),
        'Random Forest': (
            Pipeline([
                ('preprocessor', preprocessor),
                ('clf', RandomForestClassifier(class_weight='balanced'))
            ]),
            {
                'clf__n_estimators': [50, 100],                         # сетка числа деревьев
                'clf__max_depth': [5, None]                             # сетка глубины
            }
        )
    }

# ——————————————————————————————————————————————————————————————
# Функция: обучает все модели через GridSearchCV и сравнивает их
# ——————————————————————————————————————————————————————————————
def train_and_compare(df, target_col):
    # 1) Разделяем таблицу на признаки X и целевую переменную y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2) Выделяем числовые фичи и создаём простой трансформер
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols)
    ])

    # 3) Кодируем строковые метки в числа 0,1,2…
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 4) Готовим кросс-валидацию
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 5) Получаем все модели и их сетки
    models = build_models(preprocessor)

    # 6) Делим данные на train/test 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    results = []
    # 7) Для каждой модели запускаем GridSearchCV
    for name, (pipeline, param_grid) in models.items():
        # GridSearchCV автоматически перебирает все сочетания параметров
        gs = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='balanced_accuracy',    # используем сбалансированную точность
            n_jobs=-1
        )
        start = time.time()
        gs.fit(X_train, y_train)           # обучаем на train
        elapsed = time.time() - start      # время обучения

        best_pipe = gs.best_estimator_     # лучший pipeline с optimal params
        y_pred = best_pipe.predict(X_test) # предсказываем на test

        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # Собираем результаты в список словарей
        results.append({
            'model': name,
            'best_params': gs.best_params_,
            'balanced_accuracy': bal_acc,
            'train_time_s': elapsed,
            'pipeline': best_pipe
        })

    # 8) Переводим в DataFrame и находим лучшую по метрике
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df['balanced_accuracy'].idxmax()]

    return results_df, best_row, le

# ——————————————————————————————————————————————————————————————
# Основная функция Streamlit-приложения
# ——————————————————————————————————————————————————————————————
def main():
    st.title("🌿 Классификация состояния растений с подбором гиперпараметров")

    # 1) Блок загрузки данных
    st.header("1. Загрузка данных")
    uploaded = st.file_uploader(
        "Выберите CSV",
        type=['csv']
    )
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(df.head(40))                  # показываем первые 5 строк
        st.session_state['data'] = df        # сохраняем в сессии

    # 2) Блок обучения + GridSearch 
    if 'data' in st.session_state:
        df = st.session_state['data']
        target = 'Состояние_здоровья_растения'


        if st.button("🚀 Обучить + Подобрать гиперпараметры"):
            with st.spinner("Идёт обучение моделей..."):
                results_df, best, le = train_and_compare(df, target)

            # Сохраняем результаты и энкодер в сессии
            st.session_state['results_df'] = results_df
            st.session_state['best']       = best
            st.session_state['le']         = le

            st.success(f"✅ Обучение завершено. Лучшая модель: {best['model']}")

    # 3) Блок отображения сравнения результатов
    if 'results_df' in st.session_state:
        st.subheader("📊 Сравнение моделей после GridSearch")
        disp = st.session_state['results_df'][[
            'model','best_params','balanced_accuracy','train_time_s'
        ]].copy()

        # Форматируем для удобства чтения
        disp['balanced_accuracy'] = disp['balanced_accuracy'].map("{:.3f}".format)
        disp['train_time_s']      = disp['train_time_s'].map("{:.2f}".format)
        disp = disp.set_index('model')

        st.table(disp)  # выводим как таблицу
        st.bar_chart(
            st.session_state['results_df']
            .set_index('model')['balanced_accuracy']
        )

        st.markdown(
            f"**🏆 Лучшая модель:** {st.session_state['best']['model']}  \n"
            # f"**Параметры:** {st.session_state['best']['best_params']}  \n"
            f"**Balanced Accuracy:** {st.session_state['best']['balanced_accuracy']:.3f}"
        )

    # 4) Блок прогнозирования лучшей моделью
    if 'best' in st.session_state:
            # Заголовок раздела на странице Streamlit
        st.header("2. Прогноз состояния растения")

        # Загружаем сохранённый датасет из сессии
        df = st.session_state['data']

        # Определяем список признаков (кроме целевой переменной)
        features = df.drop(columns=[target]).columns

        # Сохраняем минимальные и максимальные значения для каждого признака
        stats_min = df[features].min()
        stats_max = df[features].max()

        # Словарь для хранения пользовательского ввода по каждому признаку
        input_vals = {}

        # Создаём две колонки для более удобного расположения элементов ввода
        cols = st.columns(2)

        # Для каждого признака создаём числовое поле ввода
        for i, feat in enumerate(features):
            mn = float(stats_min[feat])  # минимум по признаку
            mx = float(stats_max[feat])  # максимум по признаку
            # default = (mn + mx) / 2      # значение по умолчанию — середина между min и max


            step = 0.1 

            # Добавляем числовой ввод в одну из двух колонок (чередование с помощью i % 2)
            with cols[i%2]:
                input_vals[feat] = st.number_input(
                    label     = feat,       # имя признака
                    min_value = mn,         # минимум
                    max_value = mx,         # максимум
                    # value     = default,    # значение по умолчанию
                    step      = step,       # шаг изменения
                    format    = "%.2f"      # отображение с 2 знаками после запятой
                )

        # Если пользователь нажал кнопку "Классифицировать"
        if st.button("🔍 Классифицировать"):
            inp_df = pd.DataFrame([input_vals])  # преобразуем словарь ввода в DataFrame

            # Загружаем лучший пайплайн (модель + предобработка)
            best_pipe = st.session_state['best']['pipeline']

            # Предсказываем индекс класса на основе введённых пользователем данных
            idx = best_pipe.predict(inp_df)[0]

            # Преобразуем индекс обратно в текстовую метку (название класса)
            orig = st.session_state['le'].inverse_transform([idx])[0]

            # Переводим метку на русский язык, если она есть в словаре label_map
            rus = label_map.get(orig, orig)

            # Выводим результат на экран
            st.success(f"Состояние растения: **{rus}**")

# Запускаем
if __name__ == "__main__":
    main()

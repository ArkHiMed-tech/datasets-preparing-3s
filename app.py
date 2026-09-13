import streamlit as st
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Настройка страницы
st.set_page_config(page_title="Прогнозирование диабета", layout="wide", page_icon="🩺")

# Функция для предобработки данных (в точности как в ноутбуке)
def preprocess_data(df):
    df = df.copy()
    zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
    for col in zero_as_missing:
        df[col] = df[col].replace(0, np.nan)
    
    df = df.dropna(subset=['Glucose'])
    
    imputer = KNNImputer(n_neighbors=5)
    feature_cols_impute = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df[feature_cols_impute] = imputer.fit_transform(df[feature_cols_impute])
    
    # Фильтрация выбросов
    df = df[(df['BloodPressure'] >= 20) & (df['BloodPressure'] <= 130)]
    df = df[(df['BMI'] >= 12) & (df['BMI'] <= 70)]
    df = df[df['SkinThickness'] <= 99]
    df = df[df['Insulin'] <= 800]
    
    # Инженерия признаков
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[-np.inf, 18.5, 25, 30, np.inf], labels=[0, 1, 2, 3]).astype(int)
    df['Age_Group'] = pd.cut(df['Age'], bins=[-np.inf, 30, 40, 50, np.inf], labels=[0, 1, 2, 3]).astype(int)
    df['Glucose_High'] = (df['Glucose'] >= 140).astype(int)
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1e-6)
    df['Pregnancies_per_Age'] = df['Pregnancies'] / (df['Age'] + 1e-6)
    
    return df

# Функция загрузки, предобработки и обучения модели (кешируется)
@st.cache_resource
def load_and_train_model():
    # Загружаем стандартный датасет Pima Indians для демонстрации
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=columns)
    
    df_processed = preprocess_data(df)
    
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Обучаем лучшую модель из ноутбука (LightGBM)
    model = lgb.LGBMClassifier(learning_rate=0.05, max_depth=3, n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train_scaled, y_train)
    
    # Инициализируем SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, X.columns.tolist(), X_train_scaled, explainer

# Загрузка модели и данных
with st.spinner("Загрузка данных и инициализация модели..."):
    model, scaler, feature_names, X_train_scaled, explainer = load_and_train_model()

# --- ИНТЕРФЕЙС ---
st.title("Интерактивный прогноз риска диабета")
st.markdown("Введите параметры пациента, чтобы получить прогноз и визуальную интерпретацию решения модели (на основе LightGBM).")

st.sidebar.header("Параметры пациента")

with st.sidebar.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Беременности (Pregnancies)", min_value=0, step=1, value=1)
        glucose = st.number_input("Глюкоза (Glucose)", min_value=0.0, format="%.1f", value=100.0)
        blood_pressure = st.number_input("Давление (BloodPressure)", min_value=0.0, format="%.1f", value=70.0)
        skin_thickness = st.number_input("Толщина кожи (SkinThickness)", min_value=0.0, format="%.1f", value=20.0)
    with col2:
        insulin = st.number_input("Инсулин (Insulin)", min_value=0.0, format="%.1f", value=80.0)
        bmi = st.number_input("ИМТ (BMI)", min_value=0.0, format="%.1f", value=25.0)
        dpf = st.number_input("Наследственность (DPF)", min_value=0.0, format="%.3f", value=0.5)
        age = st.number_input("Возраст (Age)", min_value=0, step=1, value=30)

    submitted = st.form_submit_button("Получить прогноз", type="primary", use_container_width=True)

if submitted:
    # 1. Подготовка входных данных
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # 2. Применяем те же преобразования, что и при обучении
    input_processed = preprocess_data(input_df)
    
    # 3. Масштабирование
    X_scaled = scaler.transform(input_processed)
    
    # 4. Предсказание
    pred_proba = model.predict_proba(X_scaled)[0]
    pred_class = model.predict(X_scaled)[0]
    
    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    st.divider()
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if pred_class == 1:
            st.error(f"**Высокий риск диабета**")
        else:
            st.success(f"**Низкий риск диабета**")
        
        st.metric(label="Вероятность диабета", value=f"{pred_proba[1]*100:.1f}%")

    with col_res2:
        st.subheader("Интерпретация решения (SHAP)")
        st.markdown("Красные факторы увеличивают риск. Синие факторы снижают риск.")
        
        try:
            # Вычисляем SHAP значения для текущего примера
            shap_values = explainer.shap_values(X_scaled)
            
            # Для бинарной классификации LGBM shap_values может быть списком [shap_neg, shap_pos]. Берем позитивный класс (индекс 1)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Создаем объект Explanation для waterfall plot
            explanation = shap.Explanation(values=shap_values[0], 
                                           base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1], 
                                           data=X_scaled[0], 
                                           feature_names=feature_names)
            
            # Визуализация в Streamlit
            st.pyplot(shap.plots.waterfall(explanation, max_display=10))
            
        except Exception as e:
            st.warning(f"Не удалось построить SHAP-визуализацию: {e}")
            
        st.caption("Диаграмма показывает, как каждый признак сдвигает базовую вероятность (base value) к итоговому предсказанию (output value).")

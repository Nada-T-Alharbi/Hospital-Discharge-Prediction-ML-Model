import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. إعداد الصفحة
# ==========================================
st.set_page_config(
    page_title="نظام توقع الخروج الذكي (Live)",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 10px; }
    div[data-testid="stMetricValue"] { font-size: 20px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. تحميل الموارد (المودل + البيانات الخام)
# ==========================================
@st.cache_resource
def load_resources():
    # تحميل المودل
    if os.path.exists("discharge_prediction_model.pkl"):
        model = joblib.load("discharge_prediction_model.pkl")
    else:
        st.error("ملف المودل غير موجود! تأكد من تشغيل كود التدريب أولاً.")
        return None, None, None, None, None

    # تحميل البيانات الخام
    try:
        patients = pd.read_csv("Dataset/hosp/patients.csv")
        admissions = pd.read_csv("Dataset/hosp/admissions.csv")
        chartevents = pd.read_csv("Dataset/icu/chartevents.csv")
        labevents = pd.read_csv("Dataset/hosp/labevents.csv")

        # تحويل التواريخ المهمة
        chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
        labevents["charttime"] = pd.to_datetime(labevents["charttime"])
        admissions["admittime"] = pd.to_datetime(admissions["admittime"])

        return model, patients, admissions, chartevents, labevents
    except FileNotFoundError as e:
        st.error(f"خطأ في تحميل البيانات: {e}")
        return None, None, None, None, None

# تحميل كل شيء مرة واحدة
model_pipeline, patients_df, admissions_df, chartevents_df, labevents_df = load_resources()

# ==========================================
# 3. دوال تجهيز بيانات المريض (Feature Extraction)
# ==========================================
def get_patient_live_data(subject_id, hadm_id):
    """تستخرج بيانات حقيقية للمريض وتعالجها بنفس طريقة تدريب المودل"""

    # 1. البيانات الديموغرافية
    pat_info = patients_df[patients_df["subject_id"] == subject_id].iloc[0]
    adm_info = admissions_df[admissions_df["hadm_id"] == hadm_id].iloc[0]

    # تحديد وقت الـ Snapshot (بعد 24 ساعة)
    t0 = adm_info["admittime"] + pd.Timedelta(hours=24)

    # 2. استخراج العلامات الحيوية (Vitals)
    VITAL_ITEMS = {
        "heart_rate": [220045], "sbp": [220179],
        "resp_rate": [220210], "spo2": [220277]
    }

    vitals_data = {}
    for name, ids in VITAL_ITEMS.items():
        subset = chartevents_df[
            (chartevents_df["hadm_id"] == hadm_id) &
            (chartevents_df["itemid"].isin(ids)) &
            (chartevents_df["charttime"] <= t0)
        ]

        if not subset.empty:
            vitals_data[f"{name}_mean"] = subset["valuenum"].mean()
            vitals_data[f"{name}_min"] = subset["valuenum"].min()
            vitals_data[f"{name}_max"] = subset["valuenum"].max()
        else:
            vitals_data[f"{name}_mean"] = np.nan
            vitals_data[f"{name}_min"] = np.nan
            vitals_data[f"{name}_max"] = np.nan

    # 3. استخراج التحاليل (Labs)
    LAB_ITEMS = {
        "creatinine": [50912], "wbc": [51300],
        "hemoglobin": [51222], "sodium": [50983]
    }

    labs_data = {}
    for name, ids in LAB_ITEMS.items():
        subset = labevents_df[
            (labevents_df["hadm_id"] == hadm_id) &
            (labevents_df["itemid"].isin(ids)) &
            (labevents_df["charttime"] <= t0)
        ]

        if not subset.empty:
            last_val = subset.sort_values("charttime").iloc[-1]["valuenum"]
            labs_data[f"{name}_last"] = last_val
        else:
            labs_data[f"{name}_last"] = np.nan

    # 4. تجميع الصف (Row) للمودل
    row = {
        "anchor_age": pat_info["anchor_age"],
        "gender": 1 if pat_info["gender"] == "M" else 0,
    }
    row.update(vitals_data)
    row.update(labs_data)

    return row, adm_info["admission_type"], pat_info["gender"]

def predict_with_model(row_data, admission_type):
    input_df = pd.DataFrame([row_data])

    # إضافة الـ Missing Indicators
    cols_to_check = [c for c in input_df.columns if "mean" in c or "last" in c or "max" in c or "min" in c]
    for col in cols_to_check:
        input_df[f"{col}_missing"] = input_df[col].isnull().astype(int)

    # إضافة عمود الـ admission_type الأصلي
    input_df["admission_type"] = admission_type

    # سوي get_dummies
    input_df = pd.get_dummies(input_df, columns=["admission_type"], prefix="admission_type")

    # مطابقة الأعمدة مع المودل
    if hasattr(model_pipeline.named_steps['scaler'], 'feature_names_in_'):
        required_cols = model_pipeline.named_steps['scaler'].feature_names_in_
        input_df = input_df.reindex(columns=required_cols, fill_value=0)

    # تعبئة القيم المفقودة
    input_df = input_df.fillna(input_df.median())
    input_df = input_df.fillna(0)

    # التوقع
    prob = model_pipeline.predict_proba(input_df)[0][1]

    # استخراج أهم العوامل
    coefs = model_pipeline.named_steps['model'].coef_[0]
    if len(coefs) == len(input_df.columns):
        contributions = input_df.values[0] * coefs
        contrib_df = pd.DataFrame({
            'Feature': input_df.columns,
            'Contribution': contributions
        }).sort_values(by='Contribution', key=abs, ascending=False).head(5)
    else:
        contrib_df = pd.DataFrame()

    return prob, contrib_df, input_df

# ==========================================
# 4. الواجهة الجانبية (Sidebar)
# ==========================================
st.sidebar.title("لوحة التحكم")
st.sidebar.header("بحث عن مريض")

if admissions_df is not None:
    available_subjects = admissions_df['subject_id'].unique()
    patient_id = st.sidebar.selectbox("اختر رقم الملف (Subject ID)", available_subjects)

    hadm_ids = admissions_df[admissions_df['subject_id'] == patient_id]['hadm_id'].unique()
    hadm_id = st.sidebar.selectbox("اختر رقم الدخول (HADM ID)", hadm_ids)

    run_analysis = st.sidebar.button("تحليل البيانات الحقيقية 🔍")
else:
    st.error("البيانات غير محملة.")
    run_analysis = False

# ==========================================
# 5. العرض الرئيسي (Main Dashboard)
# ==========================================
if run_analysis and model_pipeline is not None:
    with st.spinner('جاري معالجة البيانات وتشغيل المودل...'):

        row_data, adm_type, gender_str = get_patient_live_data(patient_id, hadm_id)
        prob, contrib_df, input_df_scaled = predict_with_model(row_data, adm_type)

        col1, col2 = st.columns([1, 3])
        with col2:
            st.title(f"ملف المريض: {patient_id}")
            st.caption(f"رقم الدخول: {hadm_id} | وقت التحليل: {pd.Timestamp.now().strftime('%H:%M')}")
            st.markdown(f"**العمر:** {row_data['anchor_age']} | **الجنس:** {gender_str} | **نوع الدخول:** {adm_type}")

        st.divider()

        # النتائج
        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("📊 نتيجة النموذج (Real-Time)")
            prob_percent = int(prob * 100)

            if prob > 0.6:
                color = "#28a745"
                status = "مرشح للخروج (Discharge)"
                box_color = "#d4edda"
            elif prob > 0.3:
                color = "#ffc107"
                status = "تحت الملاحظة (Observation)"
                box_color = "#fff3cd"
            else:
                color = "#dc3545"
                status = "يتطلب بقاء (Stay)"
                box_color = "#f8d7da"

            st.markdown(f"""
            <div style="background-color: {box_color}; padding: 20px; border-radius: 15px; border-left: 10px solid {color}; text-align: center;">
                <h1 style="color: {color}; font-size: 50px; margin:0;">{prob_percent}%</h1>
                <h3 style="margin:0; color: #333;">احتمالية الخروج خلال 48 ساعة</h3>
                <p style="font-size: 18px; font-weight: bold; margin-top: 10px;">التوصية: {status}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.info("ℹ️ هذه النتيجة مبنية على بيانات MIMIC-IV الحقيقية المحملة في النظام.")

        st.divider()

        # المؤشرات
        st.subheader("📈 القيم المدخلة للمودل")
        m1, m2, m3, m4 = st.columns(4)

        def fmt(val): return f"{val:.1f}" if pd.notnull(val) else "N/A"

        m1.metric("Avg HR", fmt(row_data.get('heart_rate_mean')))
        m2.metric("Avg SpO2", fmt(row_data.get('spo2_mean')))
        m3.metric("Last WBC", fmt(row_data.get('wbc_last')))
        m4.metric("Last Creatinine", fmt(row_data.get('creatinine_last')))

        # التفسير - Coefficients
        if not contrib_df.empty:
            st.subheader("🔧 التأثير المباشر للعوامل (Feature Coefficients)")
            st.bar_chart(contrib_df.set_index('Feature')['Contribution'])
            st.caption("القيم الموجبة تدفع نحو الخروج، السالبة تدفع نحو البقاء.")
        else:
            st.warning("لم يتمكن النظام من حساب تأثير العوامل بدقة لهذا المريض.")

        # SHAP Explanation
        st.divider()
        st.subheader("🧠 شرح النموذج باستخدام SHAP")
        
        try:
            # تحميل SHAP data إذا كانت متوفرة
            if os.path.exists("shap_values_data.pkl"):
                shap_data = joblib.load("shap_values_data.pkl")
                
                # إعادة بناء البيانات المحفوظة
                shap_values_test = shap_data['shap_values']
                X_test_scaled = pd.DataFrame(
                    shap_data['X_test_scaled'],
                    columns=shap_data['X_test_scaled_columns']
                )
                background_sample = pd.DataFrame(
                    shap_data['background_sample'],
                    columns=shap_data['background_columns']
                )
                expected_value = shap_data['expected_value']
                
                # إعادة إنشاء SHAP explainer للمريض الحالي فقط
                scaler = model_pipeline.named_steps['scaler']
                input_scaled = scaler.transform(input_df_scaled.values)
                input_scaled_df = pd.DataFrame(input_scaled, columns=input_df_scaled.columns)
                
                # استخدم KernelExplainer للحصول على SHAP values للمريض الحالي
                current_explainer = shap.KernelExplainer(
                    model_pipeline.named_steps['model'].predict,
                    background_sample
                )
                shap_value_patient = current_explainer.shap_values(input_scaled_df)
                
                st.info("💡 **SHAP يشرح كيف يتنبأ النموذج:**")
                st.write("""
                - كل ميزة لها تأثير معين على التنبؤ
                - الأحمر: يزيد احتمالية الخروج
                - الأزرق: يقلل احتمالية الخروج
                - الحجم: قوة التأثير
                """)
                
                # عرض أهم المساهمات
                st.write("**أهم 5 عوامل مؤثرة على التنبؤ:**")
                shap_summary = pd.DataFrame({
                    'Feature': input_df_scaled.columns,
                    'SHAP Value': np.abs(shap_value_patient[0])
                }).nlargest(5, 'SHAP Value')
                
                st.dataframe(shap_summary, use_container_width=True)
                
                # Force plot (Text version)
                st.write("**تفصيل التنبؤ:**")
                
                positive_effects = []
                negative_effects = []
                
                for feat, shap_val in zip(input_df_scaled.columns, shap_value_patient[0]):
                    if shap_val > 0:
                        positive_effects.append((feat, shap_val))
                    else:
                        negative_effects.append((feat, abs(shap_val)))
                
                positive_effects.sort(key=lambda x: x[1], reverse=True)
                negative_effects.sort(key=lambda x: x[1], reverse=True)
                
                force_text = f"""
**التأثيرات الإيجابية (تزيد احتمالية الخروج):**
"""
                for feat, shap_val in positive_effects[:5]:
                    force_text += f"\n- {feat}: +{shap_val:.4f}"
                
                force_text += f"\n\n**التأثيرات السلبية (تقلل احتمالية الخروج):**"
                for feat, shap_val in negative_effects[:5]:
                    force_text += f"\n- {feat}: -{shap_val:.4f}"
                
                st.info(force_text)
                
            else:
                st.warning("⚠️ بيانات SHAP غير متوفرة. يرجى إعادة تدريب النموذج.")
        except Exception as e:
            st.warning(f"⚠️ خطأ في حساب SHAP: {str(e)}")
            st.info("سيتم عرض التفسيرات باستخدام Coefficients بدلاً من ذلك.")

elif not run_analysis:
    st.info("👈 اختر مريضاً من القائمة الجانبية للبدء.")

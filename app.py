import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NutriScan AI",
    page_icon="🥗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM CSS (THE DESIGN ENGINE)
# ==========================================
st.markdown("""
    <style>
    /* Import Poppins Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* General App Styling */
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        font-family: 'Poppins', sans-serif !important;
        color: #2D3436;
    }

    /* Custom Header Styling */
    .main-header {
        font-size: 24px;
        font-weight: 700;
        color: #56AB91;
        margin-bottom: 5px;
        text-align: center;
    }
    .sub-header {
        font-size: 14px;
        color: #B2BEC3;
        font-weight: 400;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Nutrient Row Styling */
    .nutrient-row {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .nutrient-box {
        background: #FFFFFF;
        border-radius: 15px;
        padding: 15px 5px;
        text-align: center;
        width: 23%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid #F0F0F0;
        transition: transform 0.2s;
    }
    .nutrient-box:hover {
        transform: translateY(-3px);
        border-color: #56AB91;
    }
    .nutrient-val {
        font-size: 16px;
        font-weight: 700;
        color: #56AB91;
    }
    .nutrient-label {
        font-size: 10px;
        color: #636E72;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Burn Row Styling */
    .burn-row {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
        gap: 10px;
    }
    .burn-box {
        background-color: #F8F9FA;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        width: 24%;
        font-size: 12px;
    }

    /* Button Styling */
    div.stButton > button {
        background-color: #56AB91;
        color: white;
        border-radius: 30px;
        padding: 12px 0px;
        font-weight: 600;
        border: none;
        width: 100%;
        box-shadow: 0 5px 15px rgba(86, 171, 145, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #469D81;
        transform: translateY(-2px);
    }

    /* Custom Radio Button (Pill Shape) */
    div[role="radiogroup"] {
        background-color: #F1F3F4;
        padding: 5px;
        border-radius: 25px;
        display: flex;
        justify-content: space-around;
    }
    div[role="radiogroup"] label {
        flex: 1;
        text-align: center;
        padding: 8px;
        border-radius: 20px;
        cursor: pointer;
        transition: 0.3s;
        border: none;
    }
    /* Hide standard radio circles */
    div[role="radiogroup"] input {
        display: none;
    }
    
    /* File Uploader Styling */
    div[data-testid="stFileUploader"] {
        border-radius: 20px;
        padding: 30px;
        border: 2px dashed #DDE1E3;
        background-color: #FAFAFA;
        text-align: center;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA & LOGIC
# ==========================================
NUTRITION_DB = {
    'chapati': {'kcal': 104, 'p': 3, 'c': 20, 'f': 1, 'unit': 'piece', 'tip': 'Whole wheat source.'},
    'dal_tadka': {'kcal': 148, 'p': 7, 'c': 18, 'f': 6, 'unit': 'bowl', 'tip': 'Protein-rich lentils.'},
    'samosa': {'kcal': 260, 'p': 4, 'c': 30, 'f': 14, 'unit': 'piece', 'tip': 'Deep fried snack.'},
    'pizza': {'kcal': 266, 'p': 11, 'c': 33, 'f': 10, 'unit': 'slice', 'tip': 'Thin crust has fewer carbs.'},
    'burger': {'kcal': 295, 'p': 17, 'c': 24, 'f': 14, 'unit': 'burger', 'tip': 'Skip mayo to save 100 kcal.'},
    'default': {'kcal': 250, 'p': 5, 'c': 30, 'f': 10, 'unit': 'serving', 'tip': 'Eat in moderation.'}
}

def get_nutrition_data(food_name):
    return NUTRITION_DB.get(food_name, NUTRITION_DB['default'])

def calculate_burn(calories):
    # Approx burn rates (kcal/min)
    return {
        "Walk": int(calories / 4.0),
        "Run": int(calories / 11.5),
        "Cycle": int(calories / 8.0),
        "Yoga": int(calories / 3.0)
    }

# Model Loading (Cached)
@st.cache_resource
def load_all_models():
    models = {}
    classes = {}
    try:
        with open('indian_food_classes.txt', 'r') as f:
            classes['indian'] = [line.strip() for line in f.readlines()]
        
        base_ind = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_ind.trainable = False
        inputs_ind = tf.keras.Input(shape=(224, 224, 3))
        x = base_ind(inputs_ind, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs_ind = tf.keras.layers.Dense(len(classes['indian']), activation='softmax')(x)
        model_ind = tf.keras.Model(inputs_ind, outputs_ind)
        model_ind.load_weights('indian_food_model.h5')
        models['indian'] = model_ind
    except: pass
    
    # Fallback mock for UI dev if models missing
    return models, classes

models, class_lists = load_all_models()

# ==========================================
# 4. UI LAYOUT
# ==========================================

# Header
st.markdown('<div class="main-header">NutriScan AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Smart Calorie & Diet Tracker</div>', unsafe_allow_html=True)

# Cuisine Switcher (Pill Style via Streamlit Radio)
mode = st.radio("Cuisine Mode", ["🇮🇳 Indian Food", "🌎 Global Food"], horizontal=True, label_visibility="collapsed")

# Image Upload
uploaded_file = st.file_uploader("Upload your meal photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. PREDICT LOGIC
    # (Using mock logic if model files missing for demo stability)
    active_key = 'indian' if "Indian" in mode else 'global'
    
    if models.get(active_key):
        active_model = models.get(active_key)
        active_classes = class_lists.get(active_key)
        
        image = Image.open(uploaded_file).convert('RGB')
        img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(img_resized)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = active_model.predict(img_array)
        idx = np.argmax(preds[0])
        food_name = active_classes[idx]
        confidence = 100 * np.max(preds[0])
    else:
        image = Image.open(uploaded_file)
        food_name = "samosa" if "Indian" in mode else "pizza"
        confidence = 92.5

    nutrition = get_nutrition_data(food_name)
    unit = nutrition['unit']

    # 2. MAIN CARD UI
    st.markdown("---")
    
    # Image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_container_width=True)
        st.markdown(f"<h3 style='text-align: center; margin: 10px 0;'>{food_name.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 12px; color: #B2BEC3;'>Confidence: {confidence:.1f}%</p>", unsafe_allow_html=True)

    # 3. DYNAMIC SLIDER
    quantity = st.slider(f"Portion Size ({unit}s)", 0.5, 5.0, 1.0, 0.5)

    # 4. REAL-TIME CALCULATIONS
    t_cal = int(nutrition['kcal'] * quantity)
    t_p = int(nutrition['p'] * quantity)
    t_c = int(nutrition['c'] * quantity)
    t_f = int(nutrition['f'] * quantity)

    # 5. NUTRIENT CARDS
    st.markdown(f"""
        <div class="nutrient-row">
            <div class="nutrient-box">
                <div class="nutrient-val">{t_c}g</div>
                <div class="nutrient-label">Carbs</div>
            </div>
            <div class="nutrient-box">
                <div class="nutrient-val">{t_p}g</div>
                <div class="nutrient-label">Protein</div>
            </div>
            <div class="nutrient-box">
                <div class="nutrient-val">{t_f}g</div>
                <div class="nutrient-label">Fats</div>
            </div>
            <div class="nutrient-box">
                <div class="nutrient-val" style="color: #FF6B6B;">{t_cal}</div>
                <div class="nutrient-label">Kcal</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 6. BURN IT OFF SECTION
    burn = calculate_burn(t_cal)
    
    if st.button("🔥 How to Burn This Off?"):
        st.markdown(f"""
            <div style="text-align: center; margin-bottom: 10px; font-weight: 600; color: #636E72;">
                To burn <span style="color: #FF6B6B;">{t_cal} kcal</span>, you need to:
            </div>
            <div class="burn-row">
                <div class="burn-box">🚶 Walk<br><b>{burn['Walk']}</b> min</div>
                <div class="burn-box">🏃 Run<br><b>{burn['Run']}</b> min</div>
                <div class="burn-box">🚴 Cycle<br><b>{burn['Cycle']}</b> min</div>
                <div class="burn-box">🧘 Yoga<br><b>{burn['Yoga']}</b> min</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.success(f"💡 **Tip:** {nutrition['tip']}")

else:
    # Empty State
    st.markdown("""
        <div style="text-align: center; margin-top: 40px; opacity: 0.6;">
            <div style="font-size: 50px;">📸</div>
            <p>Upload a meal to analyze</p>
        </div>
    """, unsafe_allow_html=True)

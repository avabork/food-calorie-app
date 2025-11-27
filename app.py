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
    layout="centered",  # Centered looks more like a mobile app
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
        background-color: #FFFFFF; /* White background like the design */
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, p, div, span {
        font-family: 'Poppins', sans-serif !important;
        color: #2D3436;
    }

    /* Custom Header Styling */
    .main-header {
        font-size: 24px;
        font-weight: 700;
        color: #56AB91; /* The Green from the image */
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 16px;
        color: #B2BEC3;
        font-weight: 400;
        margin-bottom: 30px;
    }

    /* Card/Container Styling */
    .food-card {
        background-color: #F8F9FA;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        text-align: center;
    }

    /* Nutrient Row Styling (The 4 boxes) */
    .nutrient-row {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .nutrient-box {
        background: #FFFFFF;
        border-radius: 15px;
        padding: 15px 10px;
        text-align: center;
        width: 23%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        border: 1px solid #F0F0F0;
    }
    .nutrient-val {
        font-size: 14px;
        font-weight: 700;
        color: #56AB91;
    }
    .nutrient-label {
        font-size: 10px;
        color: #636E72;
        margin-top: 5px;
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

    /* File Uploader Styling */
    div[data-testid="stFileUploader"] {
        border-radius: 20px;
        padding: 20px;
        border: 2px dashed #56AB91;
        background-color: #F0FDF4;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOGIC & DATA
# ==========================================
# [KEEPING YOUR EXISTING DB LOGIC]
NUTRITION_DB = {
    'chapati': {'kcal': 104, 'p': 3, 'c': 20, 'f': 1, 'unit': 'piece', 'tip': 'Whole wheat source.'},
    'dal_tadka': {'kcal': 148, 'p': 7, 'c': 18, 'f': 6, 'unit': 'bowl (150g)', 'tip': 'Protein-rich lentils.'},
    'samosa': {'kcal': 260, 'p': 4, 'c': 30, 'f': 14, 'unit': 'piece', 'tip': 'Deep fried snack.'},
    # ... (Your full DB goes here) ...
    'default': {'kcal': 250, 'p': 5, 'c': 30, 'f': 10, 'unit': 'serving', 'tip': 'Eat in moderation.'}
}

def get_nutrition_data(food_name):
    return NUTRITION_DB.get(food_name, NUTRITION_DB['default'])

# Helper to load models (Cached)
@st.cache_resource
def load_all_models():
    models = {}
    classes = {}
    try:
        # Load Indian Model
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
    except:
        pass # Handle gracefully if file missing
    return models, classes

models, class_lists = load_all_models()

# ==========================================
# 4. UI STRUCTURE
# ==========================================

# --- HEADER SECTION ---
st.markdown('<div class="main-header">Let\'s Check Food</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header" style="color: #2D3436;">Nutrition & Calories</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Select food image to see calories</div>', unsafe_allow_html=True)

# --- MODE SELECTION (Styled as Radio for simplicity) ---
mode = st.radio("Cuisine Type", ["🇮🇳 Indian", "🌎 Global"], horizontal=True, label_visibility="collapsed")

# --- UPLOAD SECTION ---
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. PREDICT
    active_key = 'indian' if "Indian" in mode else 'global'
    
    # Mocking prediction if files are missing for UI demo purposes
    # In real app, un-comment the model logic
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
        # Fallback for UI testing
        image = Image.open(uploaded_file)
        food_name = "samosa" # Demo value
        confidence = 95.0

    # Get Data
    nutrition = get_nutrition_data(food_name)
    
    # 2. DISPLAY RESULT (The "Right Side" of your design)
    st.markdown("---")
    
    # Centered Round Image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Circular mask implementation via st.image is hard, so we use standard rounded
        st.image(image, use_container_width=True)
    
    # Title & Description
    st.markdown(f"""
        <div style="text-align: center; margin-top: 10px;">
            <h2 style="margin:0; font-size: 28px; color: #2D3436;">{food_name.replace('_', ' ').title()}</h2>
            <p style="color: #B2BEC3; font-size: 14px;">{nutrition['tip']}</p>
        </div>
    """, unsafe_allow_html=True)

    # 3. DYNAMIC PORTION SLIDER
    quantity = st.slider("Portion Size (Pieces/Bowls)", 0.5, 5.0, 1.0, 0.5)
    
    # Calc Totals
    t_cal = int(nutrition['kcal'] * quantity)
    t_p = int(nutrition['p'] * quantity)
    t_c = int(nutrition['c'] * quantity)
    t_f = int(nutrition['f'] * quantity)

    # 4. CUSTOM NUTRIENT ROW (Matches your image)
    st.markdown(f"""
        <div class="nutrient-row">
            <div class="nutrient-box">
                <div class="nutrient-val">{t_c}g</div>
                <div class="nutrient-label">Carbs</div>
            </div>
            <div class="nutrient-box">
                <div class="nutrient-val">{t_p}g</div>
                <div class="nutrient-label">Proteins</div>
            </div>
            <div class="nutrient-box">
                <div class="nutrient-val">{t_f}g</div>
                <div class="nutrient-label">Fats</div>
            </div>
            <div class="nutrient-box">
                <div class="nutrient-val">{t_cal}</div>
                <div class="nutrient-label">Calories</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 5. ACTION BUTTON
    if st.button(f"🔥 Burn {t_cal} Calories"):
        # Burn Logic
        walk = int(t_cal / 4)
        run = int(t_cal / 11.5)
        st.success(f"🏃 Run for {run} mins or 🚶 Walk for {walk} mins to burn this off!")

else:
    # Placeholder Graphic
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; opacity: 0.5;">
            <h1 style="font-size: 60px;">📸</h1>
            <p>Snap a photo to start tracking</p>
        </div>
    """, unsafe_allow_html=True)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# ==========================================
# 1. APP CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Indian Food Calorie AI",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern dark look
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #303030;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        border: 1px solid #FF4B4B;
    }
    /* Custom Title */
    h1 {
        color: #FF4B4B;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
    }
    /* Info Box */
    .stAlert {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #4CAF50;
    }
    /* Button Styling */
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #FF2B2B;
        color: white;
        box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Hide deprecation warnings
st.set_option('client.showErrorDetails', False)

# ==========================================
# 2. NUTRITION DATABASE (The "Brain")
# ==========================================
NUTRITION_DB = {
    # --- SPECIAL REQUESTS (PRECISE) ---
    'chapati': {
        'kcal': 104, 'p': 3, 'c': 20, 'f': 1, 
        'tip': 'A standard 6-inch whole wheat chapati. Low Glycemic Index and good source of fiber.'
    },
    'dal_tadka': {
        'kcal': 148, 'p': 7, 'c': 18, 'f': 6, 
        'tip': 'Values per 100g. Yellow lentils are great protein. The "Tadka" (tempering) adds the fat content.'
    },
    # ... (Rest of your DB remains the same) ...
    'samosa': {'kcal': 260, 'p': 4, 'c': 30, 'f': 14, 'tip': 'Deep fried! Limit to 1.'},
    'naan': {'kcal': 260, 'p': 8, 'c': 45, 'f': 5, 'tip': 'Made of refined flour (Maida).'},
    'biryani': {'kcal': 200, 'p': 8, 'c': 25, 'f': 9, 'tip': 'Values per 100g. High calorie due to ghee.'},
}

# Helper to safely get nutrition info
def get_nutrition(food_name):
    default_nutrition = {
        'kcal': 250, 'p': 5, 'c': 30, 'f': 10, 
        'tip': 'A delicious Indian dish. Enjoy in moderation!'
    }
    return NUTRITION_DB.get(food_name, default_nutrition)

# ==========================================
# 3. ROBUST MODEL LOADING (The Fix)
# ==========================================
@st.cache_resource
def load_artifacts():
    # Check files first
    if not os.path.exists('indian_food_model.h5'):
        st.error("❌ 'indian_food_model.h5' not found. Please upload it.")
        return None, []
    
    if not os.path.exists('indian_food_classes.txt'):
        st.error("❌ 'indian_food_classes.txt' not found. Please upload it.")
        return None, []

    # 1. Load Class Names
    with open('indian_food_classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # 2. Rebuild & Load Weights (Fixes the "2 inputs" error)
    try:
        # Recreate the exact same architecture used in training
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Load the saved weights into this fresh structure
        model.load_weights('indian_food_model.h5')
        
    except Exception as e:
        # Fallback: Try standard load if rebuild fails (for older file versions)
        try:
            model = tf.keras.models.load_model('indian_food_model.h5')
        except Exception as e2:
            st.error(f"❌ Critical Error loading model: {e2}")
            return None, []

    return model, class_names

# Initialize
model, FOOD_CLASSES = load_artifacts()

if model is None:
    st.stop()

# ==========================================
# 4. PREDICTION ENGINE
# ==========================================
def predict_image(image_file):
    # Load and Resize
    image = Image.open(image_file).convert('RGB')
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    # Preprocess
    img_array = np.asarray(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Inference
    predictions = model.predict(img_array)
    idx = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0])
    
    return FOOD_CLASSES[idx], confidence

# ==========================================
# 5. USER INTERFACE
# ==========================================

# Sidebar for inputs
with st.sidebar:
    st.title("📸 Input")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    st.markdown("### 💡 How to use")
    st.info(
        "1. Upload a clear photo of food.\n"
        "2. Click 'Analyze Nutrition'.\n"
        "3. Get instant calorie stats!"
    )

# Main Content Area
st.title("🍛 Indian Food Calorie AI")
st.markdown("##### Smart Nutritional Tracking Powered by Deep Learning")

if uploaded_file is not None:
    # Two columns: Image on left, Results on right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption='Your Meal', use_container_width=True)
        analyze_btn = st.button("🔍 Analyze Nutrition")
    
    if analyze_btn:
        with col2:
            with st.spinner("🧠 Analyzing food texture & shape..."):
                food_name, confidence = predict_image(uploaded_file)
                nutrition = get_nutrition(food_name)
            
            # --- RESULTS SECTION ---
            st.success(f"**Detected:** {food_name.replace('_', ' ').title()}")
            st.progress(int(confidence), text=f"Confidence: {confidence:.1f}%")
            
            # Metrics Row (Custom Cards via CSS)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔥 Calories", f"{nutrition['kcal']}", "kcal")
            m2.metric("🥩 Protein", f"{nutrition['p']}g")
            m3.metric("🍞 Carbs", f"{nutrition['c']}g")
            m4.metric("🥑 Fat", f"{nutrition['f']}g")
            
            # Health Tip Box
            st.info(f"💡 **Health Tip:** {nutrition['tip']}")
            
            # Chart
            st.subheader("Macro Distribution")
            chart_data = {
                "Nutrient": ["Protein", "Carbs", "Fat"],
                "Grams": [nutrition['p'], nutrition['c'], nutrition['f']]
            }
            st.bar_chart(chart_data, x="Nutrient", y="Grams", color="#FF4B4B")
else:
    # Placeholder when no image is uploaded
    st.markdown(
        """
        <div style='text-align: center; padding: 50px; color: #555;'>
            <h3>👈 Upload an image from the sidebar to get started!</h3>
            <p>Supported formats: JPG, PNG, JPEG</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

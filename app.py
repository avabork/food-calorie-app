import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Indian Food Calorie AI",
    page_icon="🍛",
    layout="centered"
)

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
    # Instead of load_model(), we rebuild the architecture and load weights
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
st.title("🍛 AI Indian Food Calorie Scanner")
st.write("Upload a photo of your meal to instantly estimate calories and macros!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Your Upload', width=300)
    
    if st.button("🔍 Analyze Nutrition"):
        with st.spinner("Analyzing pixels..."):
            food_name, confidence = predict_image(uploaded_file)
            nutrition = get_nutrition(food_name)
        
        # --- RESULTS SECTION ---
        st.markdown(f"### 🍽️ Detected: **{food_name.replace('_', ' ').title()}**")
        st.caption(f"Confidence: {confidence:.1f}%")
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔥 Calories", f"{nutrition['kcal']}", "kcal")
        col2.metric("🥩 Protein", f"{nutrition['p']}g")
        col3.metric("🍞 Carbs", f"{nutrition['c']}g")
        col4.metric("🥑 Fat", f"{nutrition['f']}g")
        
        st.info(f"💡 **Health Tip:** {nutrition['tip']}")
        
        # Chart
        st.subheader("Macro Nutrient Profile")
        chart_data = {
            "Nutrient": ["Protein", "Carbs", "Fat"],
            "Grams": [nutrition['p'], nutrition['c'], nutrition['f']]
        }
        st.bar_chart(chart_data, x="Nutrient", y="Grams", color="#FF4B4B")

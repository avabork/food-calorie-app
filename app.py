import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# ==========================================
# 1. APP CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="AI Smart Dietician",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    
    /* Card Styling */
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
        border: 1px solid #4CAF50;
    }
    
    /* Exercise Box */
    .exercise-box {
        background-color: #1E2329;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-top: 20px;
    }
    
    h1 { color: #4CAF50; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; text-align: center; }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #4CAF50; color: white; border-radius: 20px;
        padding: 10px 24px; font-weight: bold; border: none; width: 100%;
    }
    div.stButton > button:hover {
        background-color: #45a049; box-shadow: 0px 4px 15px rgba(76, 175, 80, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

st.set_option('client.showErrorDetails', False)

# ==========================================
# 2. MASTER NUTRITION DATABASE
# ==========================================
NUTRITION_DB = {
    # --- INDIAN FOODS ---
    'chapati': {'kcal': 104, 'p': 3, 'c': 20, 'f': 1, 'unit': 'piece', 'tip': 'Standard 6-inch whole wheat.'},
    'dal_tadka': {'kcal': 148, 'p': 7, 'c': 18, 'f': 6, 'unit': 'bowl (150g)', 'tip': 'Protein-rich lentils.'},
    'dal_makhani': {'kcal': 280, 'p': 10, 'c': 25, 'f': 16, 'unit': 'bowl (150g)', 'tip': 'Rich in butter/cream.'},
    'samosa': {'kcal': 260, 'p': 4, 'c': 30, 'f': 14, 'unit': 'piece', 'tip': 'Deep fried. Limit to one!'},
    'naan': {'kcal': 260, 'p': 8, 'c': 45, 'f': 5, 'unit': 'piece', 'tip': 'Refined flour bread.'},
    'biryani': {'kcal': 200, 'p': 8, 'c': 25, 'f': 9, 'unit': 'plate (200g)', 'tip': 'Calorie dense.'},
    'daal_baati_churma': {'kcal': 450, 'p': 12, 'c': 60, 'f': 20, 'unit': 'serving', 'tip': 'High ghee content.'},
    'daal_puri': {'kcal': 210, 'p': 6, 'c': 30, 'f': 9, 'unit': 'piece', 'tip': 'Fried bread.'},
    'makki_di_roti_sarson_da_saag': {'kcal': 350, 'p': 8, 'c': 40, 'f': 18, 'unit': 'serving', 'tip': 'Winter staple.'},
    'misi_roti': {'kcal': 130, 'p': 5, 'c': 25, 'f': 2, 'unit': 'piece', 'tip': 'Gram flour bread.'},
    'poha': {'kcal': 180, 'p': 3, 'c': 35, 'f': 5, 'unit': 'bowl', 'tip': 'Light breakfast.'},
    'litti_chokha': {'kcal': 280, 'p': 10, 'c': 45, 'f': 6, 'unit': 'serving', 'tip': 'Baked wheat balls.'},
    'aloo_gobi': {'kcal': 110, 'p': 3, 'c': 15, 'f': 5, 'unit': 'bowl', 'tip': 'Healthy if low oil.'},
    'aloo_matar': {'kcal': 130, 'p': 4, 'c': 18, 'f': 6, 'unit': 'bowl', 'tip': 'Potatoes and peas.'},
    'butter_chicken': {'kcal': 280, 'p': 14, 'c': 8, 'f': 22, 'unit': 'bowl', 'tip': 'Creamy tomato gravy.'},
    'chana_masala': {'kcal': 165, 'p': 8, 'c': 25, 'f': 5, 'unit': 'bowl', 'tip': 'Chickpea curry.'},
    'chicken_tikka_masala': {'kcal': 250, 'p': 16, 'c': 10, 'f': 18, 'unit': 'bowl', 'tip': 'Spicy tomato sauce.'},
    'kadai_paneer': {'kcal': 260, 'p': 12, 'c': 8, 'f': 20, 'unit': 'bowl', 'tip': 'Paneer with bell peppers.'},
    'palak_paneer': {'kcal': 220, 'p': 10, 'c': 6, 'f': 18, 'unit': 'bowl', 'tip': 'Spinach & Paneer.'},
    'paneer_butter_masala': {'kcal': 350, 'p': 12, 'c': 15, 'f': 28, 'unit': 'bowl', 'tip': 'Rich paneer gravy.'},
    'idli': {'kcal': 39, 'p': 2, 'c': 8, 'f': 0, 'unit': 'piece', 'tip': 'Steamed rice cake.'},
    'dosa': {'kcal': 168, 'p': 4, 'c': 29, 'f': 4, 'unit': 'piece', 'tip': 'Rice crepe.'},
    'chole_bhature': {'kcal': 450, 'p': 14, 'c': 55, 'f': 20, 'unit': 'serving', 'tip': 'Heavy meal.'},
    'chai': {'kcal': 100, 'p': 2, 'c': 15, 'f': 3, 'unit': 'cup', 'tip': 'Sugar adds calories.'},
    'dhokla': {'kcal': 160, 'p': 8, 'c': 25, 'f': 5, 'unit': 'piece', 'tip': 'Steamed gram flour.'},
    'gulab_jamun': {'kcal': 150, 'p': 2, 'c': 25, 'f': 6, 'unit': 'piece', 'tip': 'Fried milk balls.'},
    'jalebi': {'kcal': 150, 'p': 0, 'c': 35, 'f': 5, 'unit': 'piece', 'tip': 'Deep fried sugar.'},
    'rasgulla': {'kcal': 120, 'p': 2, 'c': 28, 'f': 1, 'unit': 'piece', 'tip': 'Sponge balls in syrup.'},

    # --- GLOBAL FOODS ---
    'pizza': {'kcal': 266, 'p': 11, 'c': 33, 'f': 10, 'unit': 'slice', 'tip': 'Thin crust has fewer carbs.'},
    'hamburger': {'kcal': 295, 'p': 17, 'c': 24, 'f': 14, 'unit': 'burger', 'tip': 'Skip mayo to save 100 kcal.'},
    'sushi': {'kcal': 143, 'p': 4, 'c': 29, 'f': 1, 'unit': 'roll', 'tip': 'Sashimi is low carb.'},
    'steak': {'kcal': 271, 'p': 26, 'c': 0, 'f': 19, 'unit': 'steak', 'tip': 'High protein.'},
    'caesar_salad': {'kcal': 44, 'p': 3, 'c': 4, 'f': 2, 'unit': 'serving', 'tip': 'Dressing adds calories.'},
    'french_fries': {'kcal': 312, 'p': 3, 'c': 41, 'f': 15, 'unit': 'serving', 'tip': 'Baked is healthier.'},
    'ramen': {'kcal': 436, 'p': 10, 'c': 55, 'f': 18, 'unit': 'bowl', 'tip': 'Broth is high sodium.'},
}

def get_nutrition_data(food_name):
    # Default fallback
    default = {'kcal': 250, 'p': 5, 'c': 30, 'f': 10, 'unit': 'serving', 'tip': 'Enjoy in moderation.'}
    return NUTRITION_DB.get(food_name, default)

def calculate_burn(calories):
    # Burn rates (approx kcal per minute for 70kg person)
    rates = {
        "Walking (Moderate)": 4.0,
        "Running (6mph)": 11.5,
        "Cycling": 8.0,
        "Yoga": 3.0
    }
    burn_time = {}
    for activity, rate in rates.items():
        burn_time[activity] = int(calories / rate)
    return burn_time

# ==========================================
# 3. MODEL LOADING & LOGIC
# ==========================================
GLOBAL_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'tako_yaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

@st.cache_resource
def load_all_models():
    models = {}
    classes = {}
    
    # 1. Load Indian Model
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
    except:
        models['indian'] = None

    # 2. Load Global Model
    try:
        base_glob = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_glob.trainable = False
        inputs_glob = tf.keras.Input(shape=(224, 224, 3))
        x = base_glob(inputs_glob, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs_glob = tf.keras.layers.Dense(101, activation='softmax')(x)
        model_glob = tf.keras.Model(inputs_glob, outputs_glob)
        model_glob.load_weights('my_food_model_pro.h5')
        models['global'] = model_glob
        classes['global'] = GLOBAL_CLASSES
    except:
        models['global'] = None
        
    return models, classes

models, class_lists = load_all_models()

# ==========================================
# 4. UI & LOGIC
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuration")
    mode = st.radio("Cuisine Mode:", ["🇮🇳 Indian Food", "🌎 Global Food"])
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

st.title("🥗 AI Smart Dietician")

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    # 1. PREDICT
    active_key = 'indian' if "Indian" in mode else 'global'
    active_model = models.get(active_key)
    active_classes = class_lists.get(active_key)
    
    if not active_model:
        st.error(f"❌ {mode} model missing. Check files.")
        st.stop()

    image = Image.open(uploaded_file).convert('RGB')
    
    # Preprocess
    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img_resized)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Inference
    try:
        preds = active_model.predict(img_array)
        idx = np.argmax(preds[0])
        food_name = active_classes[idx]
        confidence = 100 * np.max(preds[0])
    except:
        # Fallback for shape mismatch (160x160)
        img_resized = ImageOps.fit(image, (160, 160), Image.Resampling.LANCZOS)
        img_array = np.asarray(img_resized)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        preds = active_model.predict(img_array)
        idx = np.argmax(preds[0])
        food_name = active_classes[idx]
        confidence = 100 * np.max(preds[0])
    
    # 2. GET BASE NUTRITION
    base_nutrition = get_nutrition_data(food_name)
    unit_label = base_nutrition.get('unit', 'serving')

    with col1:
        st.image(image, caption='Your Meal', use_container_width=True)
        st.success(f"**Detected:** {food_name.replace('_', ' ').title()}")
        st.progress(int(confidence), text=f"Confidence: {confidence:.1f}%")

    with col2:
        st.markdown("### 🍽️ Portion Controller")
        
        # 3. DYNAMIC PORTION SLIDER
        quantity = st.number_input(
            f"How many {unit_label}s?", 
            min_value=0.5, 
            max_value=10.0, 
            value=1.0, 
            step=0.5
        )
        
        # 4. CALCULATE TOTALS
        total_kcal = int(base_nutrition['kcal'] * quantity)
        total_p = round(base_nutrition['p'] * quantity, 1)
        total_c = round(base_nutrition['c'] * quantity, 1)
        total_f = round(base_nutrition['f'] * quantity, 1)

        # Display Totals
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔥 Calories", f"{total_kcal}")
        m2.metric("🥩 Protein", f"{total_p}g")
        m3.metric("🍞 Carbs", f"{total_c}g")
        m4.metric("🥑 Fat", f"{total_f}g")
        
        st.info(f"💡 **Tip:** {base_nutrition['tip']}")
        
        chart_data = {"Nutrient": ["Protein", "Carbs", "Fat"], "Grams": [total_p, total_c, total_f]}
        st.bar_chart(chart_data, x="Nutrient", y="Grams", color="#4CAF50")

    # ==========================================
    # 5. NEW: EXERCISE BURN ESTIMATOR
    # ==========================================
    st.markdown("---")
    st.subheader("🔥 Burn It Off!")
    
    burn_times = calculate_burn(total_kcal)
    
    e1, e2, e3, e4 = st.columns(4)
    
    e1.markdown(f"**🚶 Walking**\n# {burn_times['Walking (Moderate)']} min")
    e2.markdown(f"**🏃 Running**\n# {burn_times['Running (6mph)']} min")
    e3.markdown(f"**🚴 Cycling**\n# {burn_times['Cycling']} min")
    e4.markdown(f"**🧘 Yoga**\n# {burn_times['Yoga']} min")

else:
    st.markdown("<div style='text-align: center; padding: 50px; color: #555;'><h3>👈 Upload an image to start!</h3></div>", unsafe_allow_html=True)

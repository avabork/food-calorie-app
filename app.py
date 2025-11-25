import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Calorie Scanner",
    page_icon="🍎",
    layout="centered"
)

# Hide warnings


# ==========================================
# 2. LOAD MODEL & DB
# ==========================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_food_model_pro.h5')
    return model

# Load the model once (cached for speed)
try:
    with st.spinner("Loading AI Brain..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# The 101 Food Classes (Must match training exactly)
FOOD_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry',
    'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse',
    'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee',
    'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings',
    'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon',
    'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup',
    'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
    'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
    'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings',
    'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
    'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
    'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'tako_yaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

NUTRITION_DB = {
    'ramen': {'kcal': 436, 'p': 10, 'c': 55, 'f': 18, 'tip': 'Broth is high in sodium. Drink water!'},
    'hummus': {'kcal': 166, 'p': 8, 'c': 14, 'f': 10, 'tip': 'Healthy fats! Great with carrots.'},
    'pizza': {'kcal': 266, 'p': 11, 'c': 33, 'f': 10, 'tip': 'Thin crust has fewer carbs.'},
    'hamburger': {'kcal': 295, 'p': 17, 'c': 24, 'f': 14, 'tip': 'Skip the mayo to save 100 kcal.'},
    'sushi': {'kcal': 143, 'p': 4, 'c': 29, 'f': 1, 'tip': 'Sashimi is a lower-carb option.'},
    'steak': {'kcal': 271, 'p': 26, 'c': 0, 'f': 19, 'tip': 'High protein, good for muscle building.'},
    'caesar_salad': {'kcal': 44, 'p': 3, 'c': 4, 'f': 2, 'tip': 'Watch out! The dressing adds most calories.'},
    'french_fries': {'kcal': 312, 'p': 3, 'c': 41, 'f': 15, 'tip': 'Baked fries are healthier than fried.'},
    'fried_rice': {'kcal': 163, 'p': 3, 'c': 32, 'f': 2, 'tip': 'Use brown rice for more fiber.'},
    'dumplings': {'kcal': 112, 'p': 4, 'c': 21, 'f': 1, 'tip': 'Steamed dumplings are lower calorie.'}
}

def get_nutrition(food_name):
    default = {'kcal': 250, 'p': 10, 'c': 30, 'f': 10, 'tip': 'Moderation is key!'}
    return NUTRITION_DB.get(food_name, default)

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
def predict(image):
    # Resize to 224x224
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Preprocess (Scale to -1..1)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    idx = np.argmax(score)
    
    return FOOD_CLASSES[idx], 100 * np.max(score)

# ==========================================
# 4. APP INTERFACE
# ==========================================
st.title("🍎 AI Food Calorie Scanner")
st.write("Upload a photo of your meal to get nutritional info instantly!")

file = st.file_uploader("Choose a food image...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Your Upload', use_column_width=True)
    
    if st.button("🔍 Analyze Calories"):
        with st.spinner("Analyzing pixels..."):
            food_name, confidence = predict(image)
            data = get_nutrition(food_name)
        
        # Display Results
        st.success(f"Detected: **{food_name.replace('_', ' ').title()}** ({confidence:.1f}%)")
        
        # Columns for Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Calories", f"{data['kcal']} kcal")
        col2.metric("Protein", f"{data['p']}g")
        col3.metric("Carbs", f"{data['c']}g")
        
        st.info(f"💡 **Health Tip:** {data['tip']}")
        
        # Macro Chart
        chart_data = {
            "Nutrient": ["Protein", "Carbs", "Fat"],
            "Grams": [data['p'], data['c'], data['f']]
        }

        st.bar_chart(chart_data, x="Nutrient", y="Grams", color="#4CAF50")

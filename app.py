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

# Custom CSS for Professional UI
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #0E1117;
    }
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #262730;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #383A42;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border: 1px solid #4CAF50;
        box-shadow: 0 8px 15px rgba(76, 175, 80, 0.2);
    }
    
    /* Headers */
    h1 {
        color: #4CAF50;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 30px;
    }
    h2, h3 {
        color: #E0E0E0;
        font-weight: 600;
    }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
        transform: scale(1.02);
    }
    
    /* Info Box Styling */
    .stAlert {
        background-color: #1E2329;
        border: 1px solid #4CAF50;
        border-radius: 8px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_option('client.showErrorDetails', False)

# ==========================================
# 2. MASTER NUTRITION DATABASE (Merged)
# ==========================================
NUTRITION_DB = {
    # --- SPECIAL REQUESTS (PRECISE) ---
    'chapati': {'kcal': 104, 'p': 3, 'c': 20, 'f': 1, 'tip': 'Standard 6-inch whole wheat. Good fiber.'},
    'dal_tadka': {'kcal': 148, 'p': 7, 'c': 18, 'f': 6, 'tip': 'Lentils with tempering. High protein.'},
    'dal_makhani': {'kcal': 280, 'p': 10, 'c': 25, 'f': 16, 'tip': 'Rich in butter/cream. Eat moderately.'},
    
    # --- INDIAN BREADS & RICE ---
    'naan': {'kcal': 260, 'p': 8, 'c': 45, 'f': 5, 'tip': 'Refined flour bread. High glycemic index.'},
    'bhatura': {'kcal': 290, 'p': 7, 'c': 40, 'f': 12, 'tip': 'Deep fried. Very calorie dense.'},
    'biryani': {'kcal': 200, 'p': 8, 'c': 25, 'f': 9, 'tip': 'Values per 100g. Rich in spices and fat.'},
    'daal_baati_churma': {'kcal': 450, 'p': 12, 'c': 60, 'f': 20, 'tip': 'Rajasthani classic. High ghee content.'},
    'daal_puri': {'kcal': 210, 'p': 6, 'c': 30, 'f': 9, 'tip': 'Fried bread stuffed with lentils.'},
    'makki_di_roti_sarson_da_saag': {'kcal': 350, 'p': 8, 'c': 40, 'f': 18, 'tip': 'Corn bread with greens. Winter staple.'},
    'misi_roti': {'kcal': 130, 'p': 5, 'c': 25, 'f': 2, 'tip': 'Made with gram flour. Good protein.'},
    'poha': {'kcal': 180, 'p': 3, 'c': 35, 'f': 5, 'tip': 'Flattened rice. Light breakfast.'},
    'litti_chokha': {'kcal': 280, 'p': 10, 'c': 45, 'f': 6, 'tip': 'Baked wheat balls with sattu stuffing.'},

    # --- INDIAN CURRIES ---
    'aloo_gobi': {'kcal': 110, 'p': 3, 'c': 15, 'f': 5, 'tip': 'Cauliflower & potato. Healthy if low oil.'},
    'aloo_matar': {'kcal': 130, 'p': 4, 'c': 18, 'f': 6, 'tip': 'Potatoes and peas curry.'},
    'aloo_methi': {'kcal': 120, 'p': 3, 'c': 16, 'f': 5, 'tip': 'Fenugreek leaves help blood sugar.'},
    'aloo_shimla_mirch': {'kcal': 125, 'p': 3, 'c': 18, 'f': 5, 'tip': 'Potato and capsicum stir fry.'},
    'bhindi_masala': {'kcal': 140, 'p': 3, 'c': 12, 'f': 9, 'tip': 'Okra curry. Low calorie vegetable.'},
    'butter_chicken': {'kcal': 280, 'p': 14, 'c': 8, 'f': 22, 'tip': 'Creamy tomato gravy. High fat.'},
    'chana_masala': {'kcal': 165, 'p': 8, 'c': 25, 'f': 5, 'tip': 'Chickpea curry. High fiber & protein.'},
    'chicken_razala': {'kcal': 220, 'p': 18, 'c': 6, 'f': 14, 'tip': 'White gravy chicken with yogurt.'},
    'chicken_tikka_masala': {'kcal': 250, 'p': 16, 'c': 10, 'f': 18, 'tip': 'Grilled chicken in spicy sauce.'},
    'dum_aloo': {'kcal': 190, 'p': 3, 'c': 22, 'f': 10, 'tip': 'Baby potatoes in spicy gravy.'},
    'kadai_paneer': {'kcal': 260, 'p': 12, 'c': 8, 'f': 20, 'tip': 'Paneer with bell peppers. High fat.'},
    'kadhi_pakoda': {'kcal': 150, 'p': 5, 'c': 12, 'f': 9, 'tip': 'Yogurt curry with fritters.'},
    'karela_bharta': {'kcal': 110, 'p': 3, 'c': 10, 'f': 7, 'tip': 'Mashed bitter gourd. Great for health.'},
    'kofta': {'kcal': 230, 'p': 6, 'c': 18, 'f': 16, 'tip': 'Vegetable balls in rich gravy.'},
    'maach_jhol': {'kcal': 140, 'p': 15, 'c': 5, 'f': 7, 'tip': 'Fish curry. Rich in Omega-3.'},
    'navrattan_korma': {'kcal': 200, 'p': 5, 'c': 20, 'f': 12, 'tip': 'Mixed veg in creamy sweet gravy.'},
    'palak_paneer': {'kcal': 220, 'p': 10, 'c': 6, 'f': 18, 'tip': 'Spinach & Paneer. Iron rich.'},
    'paneer_butter_masala': {'kcal': 350, 'p': 12, 'c': 15, 'f': 28, 'tip': 'Rich paneer gravy. Calorie dense.'},

    # --- INDIAN SNACKS ---
    'aloo_tikki': {'kcal': 180, 'p': 3, 'c': 25, 'f': 8, 'tip': 'Fried potato patty.'},
    'chicken_tikka': {'kcal': 150, 'p': 25, 'c': 2, 'f': 5, 'tip': 'Grilled lean chicken. Excellent macro profile.'},
    'kachori': {'kcal': 190, 'p': 4, 'c': 20, 'f': 11, 'tip': 'Deep fried stuffed pastry.'},
    'kuzhi_paniyaram': {'kcal': 60, 'p': 2, 'c': 10, 'f': 1, 'tip': 'Steamed/fried batter balls (per piece).'},
    'samosa': {'kcal': 260, 'p': 4, 'c': 30, 'f': 14, 'tip': 'Deep fried pastry. Limit intake.'},
    'unni_appam': {'kcal': 70, 'p': 1, 'c': 12, 'f': 3, 'tip': 'Sweet rice fritter (per piece).'},
    'idli': {'kcal': 39, 'p': 2, 'c': 8, 'f': 0, 'tip': 'Steamed rice cake. Very healthy.'},
    'dosa': {'kcal': 168, 'p': 4, 'c': 29, 'f': 4, 'tip': 'Rice crepe. Healthy if less oil used.'},
    'chole_bhature': {'kcal': 450, 'p': 14, 'c': 55, 'f': 20, 'tip': 'Heavy meal. High calorie & fat.'},
    'butter_naan': {'kcal': 310, 'p': 9, 'c': 48, 'f': 10, 'tip': 'Buttered refined flour bread.'},
    'chai': {'kcal': 100, 'p': 2, 'c': 15, 'f': 3, 'tip': 'Indian tea. Sugar adds calories.'},
    'dhokla': {'kcal': 160, 'p': 8, 'c': 25, 'f': 5, 'tip': 'Steamed gram flour cake. Healthy snack.'},

    # --- INDIAN SWEETS ---
    'adhirasam': {'kcal': 180, 'p': 1, 'c': 35, 'f': 6, 'tip': 'Fried jaggery sweet.'},
    'anarsa': {'kcal': 150, 'p': 2, 'c': 25, 'f': 5, 'tip': 'Rice & jaggery pastry.'},
    'ariselu': {'kcal': 160, 'p': 1, 'c': 30, 'f': 5, 'tip': 'Traditional fried sweet.'},
    'bandar_laddu': {'kcal': 200, 'p': 3, 'c': 30, 'f': 9, 'tip': 'Besan laddu.'},
    'basundi': {'kcal': 280, 'p': 8, 'c': 35, 'f': 12, 'tip': 'Thickened sweet milk.'},
    'boondi': {'kcal': 230, 'p': 2, 'c': 30, 'f': 12, 'tip': 'Sweet fried droplets.'},
    'chak_hao_kheer': {'kcal': 250, 'p': 5, 'c': 40, 'f': 8, 'tip': 'Black rice pudding.'},
    'cham_cham': {'kcal': 175, 'p': 4, 'c': 35, 'f': 2, 'tip': 'Bengali sweet.'},
    'chhena_kheeri': {'kcal': 220, 'p': 7, 'c': 30, 'f': 9, 'tip': 'Cheese pudding.'},
    'chikki': {'kcal': 120, 'p': 4, 'c': 15, 'f': 6, 'tip': 'Peanut brittle. Good energy.'},
    'dharwad_pedha': {'kcal': 130, 'p': 4, 'c': 20, 'f': 5, 'tip': 'Milk sweet.'},
    'doodhpak': {'kcal': 260, 'p': 7, 'c': 35, 'f': 10, 'tip': 'Rice pudding.'},
    'double_ka_meetha': {'kcal': 350, 'p': 5, 'c': 55, 'f': 15, 'tip': 'Fried bread pudding.'},
    'gajar_ka_halwa': {'kcal': 250, 'p': 4, 'c': 40, 'f': 10, 'tip': 'Carrot pudding with ghee.'},
    'gavvalu': {'kcal': 80, 'p': 1, 'c': 15, 'f': 3, 'tip': 'Sweet shell.'},
    'ghevar': {'kcal': 300, 'p': 3, 'c': 45, 'f': 15, 'tip': 'Soaked honeycomb pastry.'},
    'gulab_jamun': {'kcal': 150, 'p': 2, 'c': 25, 'f': 6, 'tip': 'Fried milk balls in syrup.'},
    'imarti': {'kcal': 150, 'p': 1, 'c': 30, 'f': 5, 'tip': 'Fried pretzel in syrup.'},
    'jalebi': {'kcal': 150, 'p': 0, 'c': 35, 'f': 5, 'tip': 'Deep fried sugar syrup spirals.'},
    'kajjikaya': {'kcal': 140, 'p': 2, 'c': 20, 'f': 6, 'tip': 'Coconut stuffed pastry.'},
    'kakinada_khaja': {'kcal': 200, 'p': 1, 'c': 35, 'f': 8, 'tip': 'Layered sweet.'},
    'kalakand': {'kcal': 140, 'p': 5, 'c': 15, 'f': 8, 'tip': 'Milk cake.'},
    'lassi': {'kcal': 150, 'p': 6, 'c': 20, 'f': 6, 'tip': 'Sweet yogurt drink.'},
    'ledikeni': {'kcal': 160, 'p': 3, 'c': 30, 'f': 4, 'tip': 'Similar to Gulab Jamun.'},
    'lyangcha': {'kcal': 180, 'p': 3, 'c': 35, 'f': 5, 'tip': 'Elongated sweet.'},
    'malapua': {'kcal': 200, 'p': 3, 'c': 30, 'f': 8, 'tip': 'Fried pancake.'},
    'misti_doi': {'kcal': 180, 'p': 6, 'c': 25, 'f': 7, 'tip': 'Sweet fermented yogurt.'},
    'modak': {'kcal': 120, 'p': 2, 'c': 20, 'f': 4, 'tip': 'Steamed sweet dumpling.'},
    'mysore_pak': {'kcal': 250, 'p': 2, 'c': 30, 'f': 15, 'tip': 'Ghee-rich sweet.'},
    'phirni': {'kcal': 220, 'p': 5, 'c': 35, 'f': 8, 'tip': 'Ground rice pudding.'},
    'pithe': {'kcal': 100, 'p': 2, 'c': 20, 'f': 1, 'tip': 'Rice cake.'},
    'poornalu': {'kcal': 150, 'p': 3, 'c': 25, 'f': 5, 'tip': 'Stuffed sweet.'},
    'pootharekulu': {'kcal': 130, 'p': 1, 'c': 25, 'f': 4, 'tip': 'Paper sweet.'},
    'qubani_ka_meetha': {'kcal': 220, 'p': 2, 'c': 50, 'f': 2, 'tip': 'Apricot sweet.'},
    'rabri': {'kcal': 300, 'p': 10, 'c': 30, 'f': 18, 'tip': 'Thickened milk.'},
    'ras_malai': {'kcal': 220, 'p': 8, 'c': 25, 'f': 10, 'tip': 'Cheese in milk.'},
    'rasgulla': {'kcal': 120, 'p': 2, 'c': 28, 'f': 1, 'tip': 'Sponge balls in syrup.'},
    'sandesh': {'kcal': 100, 'p': 4, 'c': 15, 'f': 3, 'tip': 'Bengali milk sweet.'},
    'shankarpali': {'kcal': 60, 'p': 1, 'c': 8, 'f': 3, 'tip': 'Fried snack.'},
    'sheer_korma': {'kcal': 250, 'p': 6, 'c': 35, 'f': 10, 'tip': 'Vermicelli pudding.'},
    'sheera': {'kcal': 280, 'p': 3, 'c': 40, 'f': 12, 'tip': 'Semolina pudding.'},
    'shrikhand': {'kcal': 260, 'p': 6, 'c': 35, 'f': 10, 'tip': 'Strained yogurt.'},
    'sohan_halwa': {'kcal': 350, 'p': 4, 'c': 50, 'f': 15, 'tip': 'Dense sweet.'},
    'sohan_papdi': {'kcal': 270, 'p': 3, 'c': 40, 'f': 12, 'tip': 'Flaky sweet.'},
    'sutar_feni': {'kcal': 150, 'p': 1, 'c': 25, 'f': 6, 'tip': 'Shredded sweet.'},
    'kulfi': {'kcal': 200, 'p': 6, 'c': 25, 'f': 10, 'tip': 'Indian ice cream.'},

    # --- GLOBAL DISHES (Food-101) ---
    'pizza': {'kcal': 266, 'p': 11, 'c': 33, 'f': 10, 'tip': 'Thin crust has fewer carbs.'},
    'hamburger': {'kcal': 295, 'p': 17, 'c': 24, 'f': 14, 'tip': 'Skip mayo to save 100 kcal.'},
    'sushi': {'kcal': 143, 'p': 4, 'c': 29, 'f': 1, 'tip': 'Sashimi is low carb.'},
    'steak': {'kcal': 271, 'p': 26, 'c': 0, 'f': 19, 'tip': 'High protein. Lean cuts best.'},
    'caesar_salad': {'kcal': 44, 'p': 3, 'c': 4, 'f': 2, 'tip': 'Dressing adds calories.'},
    'french_fries': {'kcal': 312, 'p': 3, 'c': 41, 'f': 15, 'tip': 'Baked is healthier.'},
    'chocolate_cake': {'kcal': 371, 'p': 5, 'c': 53, 'f': 15, 'tip': 'High sugar. Share it!'},
    'ramen': {'kcal': 436, 'p': 10, 'c': 55, 'f': 18, 'tip': 'Broth is high sodium.'},
}

# Helper
def get_nutrition(food_name):
    # Default fallback
    default = {'kcal': 250, 'p': 5, 'c': 30, 'f': 10, 'tip': 'Delicious! Eat in moderation.'}
    return NUTRITION_DB.get(food_name, default)

# ==========================================
# 3. MODEL LOADING & LOGIC
# ==========================================
# Hardcoded Food-101 Classes (to save a file)
GLOBAL_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'tako_yaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

@st.cache_resource
def load_all_models():
    models = {}
    classes = {}
    
    # 1. Load Indian Model
    try:
        # Rebuild Indian Model Architecture (Safe Loading)
        with open('indian_food_classes.txt', 'r') as f:
            ind_classes = [line.strip() for line in f.readlines()]
        classes['indian'] = ind_classes
        
        base_ind = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_ind.trainable = False
        inputs_ind = tf.keras.Input(shape=(224, 224, 3))
        x = base_ind(inputs_ind, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs_ind = tf.keras.layers.Dense(len(ind_classes), activation='softmax')(x)
        model_ind = tf.keras.Model(inputs_ind, outputs_ind)
        
        model_ind.load_weights('indian_food_model.h5')
        models['indian'] = model_ind
    except Exception as e:
        st.error(f"⚠️ Indian Model Load Error: {e}")
        models['indian'] = None

    # 2. Load Global Model
    try:
        # Load directly if it was saved as a full model, otherwise rebuild
        # Attempt rebuild first as it's safer for weights-only files
        base_glob = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_glob.trainable = False
        inputs_glob = tf.keras.Input(shape=(224, 224, 3))
        x = base_glob(inputs_glob, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs_glob = tf.keras.layers.Dense(101, activation='softmax')(x)
        model_glob = tf.keras.Model(inputs_glob, outputs_glob)
        
        try:
            model_glob.load_weights('my_food_model_pro.h5')
        except ValueError:
             # Fallback for shape mismatch: try loading whole model directly
             model_glob = tf.keras.models.load_model('my_food_model_pro.h5')

        models['global'] = model_glob
        classes['global'] = GLOBAL_CLASSES
    except Exception as e:
        st.warning(f"⚠️ Global Model Load Error: {e}")
        models['global'] = None
        
    return models, classes

# Initialize
models, class_lists = load_all_models()

# ==========================================
# 4. UI & LOGIC
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # THE SWITCHER
    mode = st.radio("Choose Cuisine Mode:", ["🇮🇳 Indian Food", "🌎 Global Food"])
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    st.info("1. Select Mode\n2. Upload Photo\n3. Click Analyze")

st.title("🥗 AI Smart Dietician")

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Your Meal', use_container_width=True)
        analyze_btn = st.button("🔍 Analyze Nutrition")
        
    if analyze_btn:
        # SELECT ACTIVE MODEL
        active_key = 'indian' if "Indian" in mode else 'global'
        active_model = models.get(active_key)
        active_classes = class_lists.get(active_key)
        
        if not active_model:
            st.error(f"❌ {mode} model is not loaded correctly. Check files.")
            st.stop()

        with col2:
            with st.spinner(f"🧠 Analyzing using {mode} Model..."):
                # Preprocess
                # Handle resolution difference: Global might be 160 or 224
                # We try 224 first as it's standard MobileNetV2
                target_size = (224, 224)
                
                img_resized = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
                img_array = np.asarray(img_resized)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                try:
                    preds = active_model.predict(img_array)
                except ValueError:
                     # If 224 fails, try 160 (legacy training size)
                     img_resized = ImageOps.fit(image, (160, 160), Image.Resampling.LANCZOS)
                     img_array = np.asarray(img_resized)
                     img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                     img_array = np.expand_dims(img_array, axis=0)
                     preds = active_model.predict(img_array)

                idx = np.argmax(preds[0])
                food_name = active_classes[idx]
                confidence = 100 * np.max(preds[0])
                
                nutrition = get_nutrition(food_name)
            
            # Display
            st.success(f"**Detected:** {food_name.replace('_', ' ').title()}")
            st.progress(int(confidence), text=f"Confidence: {confidence:.1f}%")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔥 Calories", f"{nutrition['kcal']}")
            m2.metric("🥩 Protein", f"{nutrition['p']}g")
            m3.metric("🍞 Carbs", f"{nutrition['c']}g")
            m4.metric("🥑 Fat", f"{nutrition['f']}g")
            
            st.info(f"💡 **Tip:** {nutrition['tip']}")
            
            chart_data = {"Nutrient": ["Protein", "Carbs", "Fat"], "Grams": [nutrition['p'], nutrition['c'], nutrition['f']]}
            st.bar_chart(chart_data, x="Nutrient", y="Grams", color="#4CAF50")
else:
    st.markdown("<div style='text-align: center; padding: 50px; color: #555;'><h3>👈 Upload an image to start!</h3></div>", unsafe_allow_html=True)

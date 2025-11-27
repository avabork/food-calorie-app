import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

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
# Values are per serving (approximate)
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

    # --- BREADS & RICE ---
    'naan': {'kcal': 260, 'p': 8, 'c': 45, 'f': 5, 'tip': 'Made of refined flour (Maida). Higher calorie than Roti.'},
    'bhatura': {'kcal': 290, 'p': 7, 'c': 40, 'f': 12, 'tip': 'Deep fried bread. Very calorie dense.'},
    'biryani': {'kcal': 200, 'p': 8, 'c': 25, 'f': 9, 'tip': 'Values per 100g. High calorie due to ghee and meat.'},
    'daal_baati_churma': {'kcal': 450, 'p': 12, 'c': 60, 'f': 20, 'tip': 'Traditional Rajasthani dish. High in ghee.'},
    'daal_puri': {'kcal': 210, 'p': 6, 'c': 30, 'f': 9, 'tip': 'Stuffed fried bread.'},
    'makki_di_roti_sarson_da_saag': {'kcal': 350, 'p': 8, 'c': 40, 'f': 18, 'tip': 'Corn flour flatbread with leafy greens. High in iron.'},
    'misi_roti': {'kcal': 130, 'p': 5, 'c': 25, 'f': 2, 'tip': 'Gram flour bread. Good protein source.'},
    'poha': {'kcal': 180, 'p': 3, 'c': 35, 'f': 5, 'tip': 'Flattened rice. Light and healthy breakfast option.'},
    'khichdi': {'kcal': 120, 'p': 4, 'c': 20, 'f': 3, 'tip': 'Easy to digest comfort food.'},
    'litti_chokha': {'kcal': 280, 'p': 10, 'c': 45, 'f': 6, 'tip': 'Baked whole wheat balls stuffed with sattu.'},

    # --- CURRIES & MAINS ---
    'aloo_gobi': {'kcal': 110, 'p': 3, 'c': 15, 'f': 5, 'tip': 'Potato and cauliflower. Healthy if less oil is used.'},
    'aloo_matar': {'kcal': 130, 'p': 4, 'c': 18, 'f': 6, 'tip': 'Peas add protein and fiber.'},
    'aloo_methi': {'kcal': 120, 'p': 3, 'c': 16, 'f': 5, 'tip': 'Fenugreek leaves are excellent for blood sugar.'},
    'aloo_shimla_mirch': {'kcal': 125, 'p': 3, 'c': 18, 'f': 5, 'tip': 'Capsicum adds Vitamin C.'},
    'bhindi_masala': {'kcal': 140, 'p': 3, 'c': 12, 'f': 9, 'tip': 'Okra is low calorie, but oil adds up.'},
    'butter_chicken': {'kcal': 280, 'p': 14, 'c': 8, 'f': 22, 'tip': 'Very rich gravy with butter and cream.'},
    'chana_masala': {'kcal': 165, 'p': 8, 'c': 25, 'f': 5, 'tip': 'Chickpeas are a powerhouse of plant protein.'},
    'chicken_razala': {'kcal': 220, 'p': 18, 'c': 6, 'f': 14, 'tip': 'White gravy chicken, rich in yogurt/cashew.'},
    'chicken_tikka_masala': {'kcal': 250, 'p': 16, 'c': 10, 'f': 18, 'tip': 'Grilled chicken in spicy tomato sauce.'},
    'dal_makhani': {'kcal': 280, 'p': 10, 'c': 25, 'f': 16, 'tip': 'Loaded with butter and cream. Eat in moderation.'},
    'dum_aloo': {'kcal': 190, 'p': 3, 'c': 22, 'f': 10, 'tip': 'Deep fried potatoes in gravy.'},
    'kadai_paneer': {'kcal': 260, 'p': 12, 'c': 8, 'f': 20, 'tip': 'High fat content from Paneer.'},
    'kadhi_pakoda': {'kcal': 150, 'p': 5, 'c': 12, 'f': 9, 'tip': 'Yogurt based curry with fried dumplings.'},
    'karela_bharta': {'kcal': 110, 'p': 3, 'c': 10, 'f': 7, 'tip': 'Bitter gourd is excellent for diabetics.'},
    'kofta': {'kcal': 230, 'p': 6, 'c': 18, 'f': 16, 'tip': 'Fried balls in gravy.'},
    'maach_jhol': {'kcal': 140, 'p': 15, 'c': 5, 'f': 7, 'tip': 'Fish curry. Good source of Omega-3.'},
    'navrattan_korma': {'kcal': 200, 'p': 5, 'c': 20, 'f': 12, 'tip': 'Mixed vegetables in creamy sweet gravy.'},
    'palak_paneer': {'kcal': 220, 'p': 10, 'c': 6, 'f': 18, 'tip': 'Spinach offers iron, paneer offers calcium.'},
    'paneer_butter_masala': {'kcal': 350, 'p': 12, 'c': 15, 'f': 28, 'tip': 'Very calorie dense. Portion control recommended.'},
    
    # --- SNACKS & STARTERS ---
    'aloo_tikki': {'kcal': 180, 'p': 3, 'c': 25, 'f': 8, 'tip': 'Fried potato patty. Often high sodium.'},
    'chicken_tikka': {'kcal': 150, 'p': 25, 'c': 2, 'f': 5, 'tip': 'Grilled and lean. Excellent protein source.'},
    'kachori': {'kcal': 190, 'p': 4, 'c': 20, 'f': 11, 'tip': 'Deep fried pastry.'},
    'kuzhi_paniyaram': {'kcal': 60, 'p': 2, 'c': 10, 'f': 1, 'tip': 'Per piece. Made from fermented batter.'},
    'unni_appam': {'kcal': 70, 'p': 1, 'c': 12, 'f': 3, 'tip': 'Sweet fritter made of rice and jaggery.'},

    # --- SWEETS & DESSERTS (Per Piece/Serving) ---
    'adhirasam': {'kcal': 180, 'p': 1, 'c': 35, 'f': 6, 'tip': 'Fried rice flour and jaggery donut.'},
    'anarsa': {'kcal': 150, 'p': 2, 'c': 25, 'f': 5, 'tip': 'Rice flour and jaggery pastry.'},
    'ariselu': {'kcal': 160, 'p': 1, 'c': 30, 'f': 5, 'tip': 'Traditional sweet, similar to Adhirasam.'},
    'bandar_laddu': {'kcal': 200, 'p': 3, 'c': 30, 'f': 9, 'tip': 'Gram flour ball with sugar and ghee.'},
    'basundi': {'kcal': 280, 'p': 8, 'c': 35, 'f': 12, 'tip': 'Sweetened condensed milk. High sugar.'},
    'boondi': {'kcal': 230, 'p': 2, 'c': 30, 'f': 12, 'tip': 'Deep fried chickpea droplets.'},
    'chak_hao_kheer': {'kcal': 250, 'p': 5, 'c': 40, 'f': 8, 'tip': 'Black rice pudding. Antioxidant rich.'},
    'cham_cham': {'kcal': 175, 'p': 4, 'c': 35, 'f': 2, 'tip': 'Milk solid sweet, similar to Rasgulla.'},
    'chhena_kheeri': {'kcal': 220, 'p': 7, 'c': 30, 'f': 9, 'tip': 'Cheese curd pudding.'},
    'chikki': {'kcal': 120, 'p': 4, 'c': 15, 'f': 6, 'tip': 'Peanut brittle. Good energy, but high sugar.'},
    'dharwad_pedha': {'kcal': 130, 'p': 4, 'c': 20, 'f': 5, 'tip': 'Caramelized milk sweet.'},
    'doodhpak': {'kcal': 260, 'p': 7, 'c': 35, 'f': 10, 'tip': 'Rice pudding with nuts.'},
    'double_ka_meetha': {'kcal': 350, 'p': 5, 'c': 55, 'f': 15, 'tip': 'Fried bread pudding. Very rich.'},
    'gajar_ka_halwa': {'kcal': 250, 'p': 4, 'c': 40, 'f': 10, 'tip': 'Carrots are healthy, but this has lots of sugar/ghee.'},
    'gavvalu': {'kcal': 80, 'p': 1, 'c': 15, 'f': 3, 'tip': 'Shell shaped flour sweet.'},
    'ghevar': {'kcal': 300, 'p': 3, 'c': 45, 'f': 15, 'tip': 'Honeycomb pastry soaked in syrup.'},
    'gulab_jamun': {'kcal': 150, 'p': 2, 'c': 25, 'f': 6, 'tip': 'Per piece. Fried milk solids in syrup.'},
    'imarti': {'kcal': 150, 'p': 1, 'c': 30, 'f': 5, 'tip': 'Lentil flour pretzel in syrup.'},
    'jalebi': {'kcal': 150, 'p': 0, 'c': 35, 'f': 5, 'tip': 'Pure sugar energy. Zero nutrition.'},
    'kajjikaya': {'kcal': 140, 'p': 2, 'c': 20, 'f': 6, 'tip': 'Coconut and sugar stuffed pastry.'},
    'kakinada_khaja': {'kcal': 200, 'p': 1, 'c': 35, 'f': 8, 'tip': 'Layered fritter soaked in syrup.'},
    'kalakand': {'kcal': 140, 'p': 5, 'c': 15, 'f': 8, 'tip': 'Milk cake. Good calcium.'},
    'lassi': {'kcal': 150, 'p': 6, 'c': 20, 'f': 6, 'tip': 'Yogurt drink. Good probiotic.'},
    'ledikeni': {'kcal': 160, 'p': 3, 'c': 30, 'f': 4, 'tip': 'Variant of Gulab Jamun.'},
    'lyangcha': {'kcal': 180, 'p': 3, 'c': 35, 'f': 5, 'tip': 'Elongated Gulab Jamun.'},
    'malapua': {'kcal': 200, 'p': 3, 'c': 30, 'f': 8, 'tip': 'Fried pancake in syrup.'},
    'misti_doi': {'kcal': 180, 'p': 6, 'c': 25, 'f': 7, 'tip': 'Fermented sweet yogurt. High sugar.'},
    'modak': {'kcal': 120, 'p': 2, 'c': 20, 'f': 4, 'tip': 'Steamed dumpling. Lower fat than fried sweets.'},
    'mysore_pak': {'kcal': 250, 'p': 2, 'c': 30, 'f': 15, 'tip': 'Very high in ghee and sugar.'},
    'phirni': {'kcal': 220, 'p': 5, 'c': 35, 'f': 8, 'tip': 'Ground rice pudding.'},
    'pithe': {'kcal': 100, 'p': 2, 'c': 20, 'f': 1, 'tip': 'Rice cake.'},
    'poornalu': {'kcal': 150, 'p': 3, 'c': 25, 'f': 5, 'tip': 'Stuffed sweet bonda.'},
    'pootharekulu': {'kcal': 130, 'p': 1, 'c': 25, 'f': 4, 'tip': 'Paper sweet. Rice starch and sugar.'},
    'qubani_ka_meetha': {'kcal': 220, 'p': 2, 'c': 50, 'f': 2, 'tip': 'Apricot dessert. Rich in fiber but sweet.'},
    'rabri': {'kcal': 300, 'p': 10, 'c': 30, 'f': 18, 'tip': 'Condensed milk with cream. High fat.'},
    'ras_malai': {'kcal': 220, 'p': 8, 'c': 25, 'f': 10, 'tip': 'Cheese balls in cream. Good calcium.'},
    'rasgulla': {'kcal': 120, 'p': 2, 'c': 28, 'f': 1, 'tip': 'Per piece. Sponge balls in syrup. Low fat.'},
    'sandesh': {'kcal': 100, 'p': 4, 'c': 15, 'f': 3, 'tip': 'Bengali sweet. Light and less sweet.'},
    'shankarpali': {'kcal': 60, 'p': 1, 'c': 8, 'f': 3, 'tip': 'Fried tea-time snack.'},
    'sheer_korma': {'kcal': 250, 'p': 6, 'c': 35, 'f': 10, 'tip': 'Vermicelli pudding with dates.'},
    'sheera': {'kcal': 280, 'p': 3, 'c': 40, 'f': 12, 'tip': 'Semolina pudding (Suji Halwa).'},
    'shrikhand': {'kcal': 260, 'p': 6, 'c': 35, 'f': 10, 'tip': 'Strained yogurt sweet.'},
    'sohan_halwa': {'kcal': 350, 'p': 4, 'c': 50, 'f': 15, 'tip': 'Dense and sticky sweet.'},
    'sohan_papdi': {'kcal': 270, 'p': 3, 'c': 40, 'f': 12, 'tip': 'Flaky sweet made of gram flour.'},
    'sutar_feni': {'kcal': 150, 'p': 1, 'c': 25, 'f': 6, 'tip': 'Shredded dough sweet.'},
}

# ==========================================
# 3. DYNAMIC MODEL LOADING
# ==========================================
@st.cache_resource
def load_artifacts():
    # 1. Load the Model
    try:
        model = tf.keras.models.load_model('indian_food_model.h5')
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, []

    # 2. Load Class Names
    try:
        with open('indian_food_classes.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        return None, []
        
    return model, class_names

# Initialize
model, FOOD_CLASSES = load_artifacts()

if model is None or not FOOD_CLASSES:
    st.warning("⚠️ Please ensure 'indian_food_model.h5' and 'indian_food_classes.txt' are in the same folder as this script.")
    st.stop()

# Helper to safely get nutrition info
def get_nutrition(food_name):
    # Default fallback if food isn't in DB yet
    default_nutrition = {
        'kcal': 250, 'p': 5, 'c': 30, 'f': 10, 
        'tip': 'A delicious Indian dish. Enjoy in moderation!'
    }
    return NUTRITION_DB.get(food_name, default_nutrition)

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
    # Display the image
    st.image(uploaded_file, caption='Your Upload', use_column_width=True)
    
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
        
        # Health Tip Box
        st.info(f"💡 **Health Tip:** {nutrition['tip']}")
        
        # Macro Distribution Chart
        st.subheader("Macro Nutrient Profile")
        chart_data = {
            "Nutrient": ["Protein", "Carbs", "Fat"],
            "Grams": [nutrition['p'], nutrition['c'], nutrition['f']]
        }
        st.bar_chart(chart_data, x="Nutrient", y="Grams", color="#FF4B4B")

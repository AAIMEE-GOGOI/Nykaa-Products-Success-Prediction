# ============================================================
# NYKAA PRODUCT SUCCESS PREDICTOR
# Brand -> Product -> ML Prediction Dashboard
# ============================================================

import os
from urllib.parse import quote_plus

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nykaa Product Success Predictor",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "nykaa_products_cleaned.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "nykaa_model.pkl"
)

BRAND_ENCODER_FILE = os.path.join(
    BASE_DIR,
    "le_brand.pkl"
)

CATEGORY_ENCODER_FILE = os.path.join(
    BASE_DIR,
    "le_cat.pkl"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   FONT
   ========================================================== */

@import url(
    'https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap'
);


/* ==========================================================
   GLOBAL
   ========================================================== */

html,
body,
[class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: #f7f8fc;
    color: #172033;
}

.block-container {
    max-width: 1500px !important;
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}


/* ==========================================================
   MAIN TITLE — FIXED
   ========================================================== */

.main-title {
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;

    color: #c72569 !important;

    font-size: 32px !important;
    line-height: 1.25 !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px !important;

    margin: 0 !important;
    padding: 8px 0 2px 0 !important;

    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;

    position: relative !important;
    z-index: 10 !important;
}

.main-subtitle {
    display: block !important;
    width: 100% !important;

    color: #667085 !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    font-weight: 500 !important;

    margin: 2px 0 25px 0 !important;
    padding: 0 !important;

    position: relative !important;
    z-index: 10 !important;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #171b2b 0%,
        #20263a 100%
    );
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.sidebar-logo {
    text-align: center;
    color: #ff4f9a !important;
    font-size: 38px;
    line-height: 1;
    font-weight: 800;
    font-style: italic;
    letter-spacing: -2px;
    margin-top: 5px;
    margin-bottom: 8px;
}

.sidebar-caption {
    text-align: center;
    color: #cbd0dc !important;
    font-size: 12px;
    line-height: 1.4;
    margin-bottom: 15px;
}

.sidebar-section {
    color: #ff5a9f !important;
    font-size: 15px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 7px;
}


/* ==========================================================
   SIDEBAR SELECTBOX
   ========================================================== */

section[data-testid="stSidebar"]
div[data-baseweb="select"] {
    background: #ffffff !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"] * {
    color: #172033 !important;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"] svg {
    fill: #172033 !important;
}


/* ==========================================================
   SIDEBAR INFO
   ========================================================== */

section[data-testid="stSidebar"]
div[data-testid="stAlert"] {
    background: #292f45 !important;
    border: 1px solid #414960 !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"]
div[data-testid="stAlert"] p {
    color: #f4f5f7 !important;
}


/* ==========================================================
   SIDEBAR PREDICTION BUTTON — FIXED
   ========================================================== */

section[data-testid="stSidebar"]
button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #ec4389,
        #f45aa0
    ) !important;

    border: none !important;

    color: #ffffff !important;

    font-weight: 700 !important;

    border-radius: 10px !important;

    min-height: 48px !important;

    font-size: 14px !important;

    box-shadow:
        0 5px 14px rgba(236, 67, 137, 0.28) !important;

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease,
        background 0.15s ease !important;
}

section[data-testid="stSidebar"]
button[kind="primary"]:hover {
    background: linear-gradient(
        135deg,
        #d92f78,
        #ec4389
    ) !important;

    color: #ffffff !important;

    transform: translateY(-1px);

    box-shadow:
        0 7px 18px rgba(236, 67, 137, 0.38) !important;
}

section[data-testid="stSidebar"]
button[kind="primary"]:active {
    transform: translateY(0px);
}


/* ==========================================================
   CARDS
   ========================================================== */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e4e7ec !important;
    border-radius: 15px !important;
    box-shadow: 0 3px 12px rgba(20, 30, 50, 0.05);
}


/* ==========================================================
   HEADINGS
   ========================================================== */

h1,
h2,
h3,
h4 {
    color: #202735 !important;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #202735 !important;
}


/* ==========================================================
   NORMAL TEXT
   ========================================================== */

.stApp p {
    color: #344054;
}

.stApp label {
    color: #344054;
}

.stCaption {
    color: #667085 !important;
}


/* ==========================================================
   PRODUCT TITLE
   ========================================================== */

.product-title {
    color: #171c28 !important;
    font-size: 21px;
    line-height: 1.35;
    font-weight: 800;
    margin-bottom: 15px;
}


/* ==========================================================
   SECTION TITLE
   ========================================================== */

.section-heading {
    color: #202735 !important;
    font-size: 20px;
    font-weight: 800;
    line-height: 1.3;
    margin-top: 8px;
    margin-bottom: 4px;
}


/* ==========================================================
   METRICS
   ========================================================== */

[data-testid="stMetricLabel"] {
    color: #667085 !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #202735 !important;
    font-weight: 800 !important;
}


/* ==========================================================
   PROGRESS BAR
   ========================================================== */

div[data-testid="stProgress"] > div > div {
    background-color: #ec4389 !important;
}


/* ==========================================================
   LINK BUTTONS
   ========================================================== */

.stLinkButton a {
    background: #ec4389 !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

.stLinkButton a:hover {
    background: #d92f78 !important;
    color: #ffffff !important;
}


/* ==========================================================
   IMAGES
   ========================================================== */

img {
    border-radius: 10px;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer-text {
    text-align: center;
    color: #ffffff !important;
    background: linear-gradient(
        90deg,
        #e83f88,
        #f477aa
    );
    padding: 11px;
    border-radius: 9px;
    margin-top: 30px;
    font-size: 12px;
    font-weight: 500;
}

.footer-text * {
    color: #ffffff !important;
}


/* ==========================================================
   RESPONSIVE TITLE
   ========================================================== */

@media (max-width: 1100px) {

    .main-title {
        font-size: 28px;
    }

}

@media (max-width: 900px) {

    .main-title {
        font-size: 25px;
    }

    .main-subtitle {
        font-size: 13px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value, default=""):
    """
    Safely convert a value to clean text.
    """

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()

    return text if text else default


def numeric(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:

        value = pd.to_numeric(
            value,
            errors="coerce"
        )

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def money(value):
    """
    Format number as Indian Rupees.
    """

    return f"₹{numeric(value):,.0f}"


def quote_url(name):
    """
    Create Nykaa search URL.
    """

    return (
        "https://www.nykaa.com/search/result/?q="
        + quote_plus(str(name))
    )


def safe_url(value, fallback_name):
    """
    Use original URL if valid.
    Otherwise create Nykaa search URL.
    """

    value = clean_text(value)

    if (
        value.startswith("http://")
        or value.startswith("https://")
    ):

        return value

    return quote_url(fallback_name)


def success_label(probability):
    """
    Convert probability into success category.
    """

    if probability >= 0.70:

        return (
            "HIGH SUCCESS",
            "success-high",
            "This product is likely to be popular."
        )

    elif probability >= 0.45:

        return (
            "MEDIUM SUCCESS",
            "success-medium",
            "This product has moderate success potential."
        )

    else:

        return (
            "LOW SUCCESS",
            "success-low",
            "This product may have lower popularity."
        )


def get_gift_flag(row):
    """
    Detect whether product text mentions a gift.
    """

    text = " ".join(
        [
            clean_text(
                row.get("tags", "")
            ),
            clean_text(
                row.get("product_title", "")
            ),
        ]
    ).lower()

    gift_words = [
        "free gift",
        "gift",
        "free comb",
        "free pouch",
        "free bag",
        "complimentary",
    ]

    return int(
        any(
            word in text
            for word in gift_words
        )
    )


def encoded_value(encoder, value):
    """
    Transform categorical value using saved encoder.
    """

    if encoder is None:
        return 0.0

    try:

        return float(
            encoder.transform(
                [
                    clean_text(value)
                ]
            )[0]
        )

    except Exception:

        return 0.0


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():

    if not os.path.exists(DATA_FILE):

        return pd.DataFrame()

    try:

        return pd.read_csv(
            DATA_FILE
        )

    except Exception as e:

        st.error(
            f"Could not read dataset: {e}"
        )

        return pd.DataFrame()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    if not os.path.exists(MODEL_FILE):

        return None

    try:

        return joblib.load(
            MODEL_FILE
        )

    except Exception:

        return None


# ============================================================
# LOAD ENCODER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_encoder(path):

    if not os.path.exists(path):

        return None

    try:

        return joblib.load(
            path
        )

    except Exception:

        return None


# ============================================================
# LOAD EVERYTHING
# ============================================================

df = load_data()

model = load_model()

brand_encoder = load_encoder(
    BRAND_ENCODER_FILE
)

category_encoder = load_encoder(
    CATEGORY_ENCODER_FILE
)


# ============================================================
# DATASET CHECK
# ============================================================

if df.empty:

    st.error(
        "❌ nykaa_products_cleaned.csv was not found."
    )

    st.info(
        "Put nykaa_products_cleaned.csv "
        "in the same folder as Nykaa.py."
    )

    st.stop()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "product_title",
    "brand_name",
    "price",
    "mrp",
    "discount_percent",
    "rating",
    "rating_count",
    "image_url",
    "product_url",
    "tags",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "Missing columns in dataset: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# NORMALISE NUMERIC DATA
# ============================================================

for column in [
    "price",
    "mrp",
    "discount_percent",
    "rating",
    "rating_count",
]:

    df[column + "__num"] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# CATEGORY
# ============================================================

if "listing_page_name" in df.columns:

    df["__category"] = (
        df["listing_page_name"]
        .fillna("Beauty")
        .astype(str)
        .str.strip()
    )

else:

    df["__category"] = "Beauty"


# ============================================================
# CLEAN PRODUCT AND BRAND
# ============================================================

df = df.reset_index(
    drop=True
)


df["__display_name"] = (
    df["product_title"]
    .fillna("Unnamed Product")
    .astype(str)
    .str.strip()
)


df["__brand_clean"] = (
    df["brand_name"]
    .fillna("Unknown Brand")
    .astype(str)
    .str.strip()
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_probability(row):

    price = numeric(
        row["price__num"]
    )

    mrp = numeric(
        row["mrp__num"],
        price
    )

    discount = numeric(
        row["discount_percent__num"]
    )

    has_gift = get_gift_flag(
        row
    )

    brand_enc = encoded_value(
        brand_encoder,
        row["brand_name"]
    )

    category_enc = encoded_value(
        category_encoder,
        row["__category"]
    )


    # ========================================================
    # EXACT MODEL FEATURE ORDER
    #
    # 1. Price
    # 2. MRP
    # 3. Discount
    # 4. Gift
    # 5. Brand
    # 6. Category
    # ========================================================

    X = np.array(
        [[
            price,
            mrp,
            discount,
            has_gift,
            brand_enc,
            category_enc,
        ]],
        dtype=float,
    )


    # ========================================================
    # RANDOM FOREST MODEL
    # ========================================================

    if model is not None:

        try:

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model
                    .predict_proba(X)[0]
                )

                classes = list(
                    getattr(
                        model,
                        "classes_",
                        [0, 1]
                    )
                )


                if 1 in classes:

                    probability = (
                        probabilities[
                            classes.index(1)
                        ]
                    )

                else:

                    probability = (
                        probabilities[-1]
                    )


                return (
                    float(
                        np.clip(
                            probability,
                            0,
                            1
                        )
                    ),
                    True,
                )


            prediction = float(
                model.predict(X)[0]
            )


            return (
                float(
                    np.clip(
                        prediction,
                        0,
                        1
                    )
                ),
                True,
            )


        except Exception:

            pass


    # ========================================================
    # FALLBACK BASELINE
    # ========================================================

    review_max = max(
        float(
            df[
                "rating_count__num"
            ].max()
        ),
        1
    )


    review_score = np.clip(
        np.log1p(
            numeric(
                row[
                    "rating_count__num"
                ]
            )
        )
        /
        np.log1p(
            review_max
        ),
        0,
        1,
    )


    discount_score = np.clip(
        discount / 60.0,
        0,
        1,
    )


    probability = (
        0.7 * review_score
        +
        0.3 * discount_score
    )


    return (
        float(
            np.clip(
                probability,
                0.05,
                0.95
            )
        ),
        False,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importances():

    default = [
        ("Brand", 27.9),
        ("Price", 27.0),
        ("MRP", 17.8),
        ("Discount", 13.8),
        ("Category", 9.4),
        ("Gift", 4.1),
    ]


    if (
        model is None
        or not hasattr(
            model,
            "feature_importances_"
        )
    ):

        return default


    try:

        values = np.asarray(
            model.feature_importances_,
            dtype=float
        )


        names = [
            "Price",
            "MRP",
            "Discount",
            "Gift",
            "Brand",
            "Category",
        ]


        if (
            len(values) != len(names)
            or values.sum() <= 0
        ):

            return default


        values = (
            values
            /
            values.sum()
            *
            100
        )


        result = list(
            zip(
                names,
                values
            )
        )


        return sorted(
            result,
            key=lambda x: x[1],
            reverse=True
        )


    except Exception:

        return default


# ============================================================
# PERCENTILE COMPARISON
# ============================================================

def percentile_info(
    value,
    series,
    higher_is_better=True
):

    clean = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()


    if clean.empty:

        return (
            "N/A",
            "No comparison data"
        )


    percentile = float(
        (
            clean <= value
        ).mean()
        * 100
    )


    if not higher_is_better:

        percentile = (
            100 - percentile
        )


    if percentile >= 75:

        return (
            "HIGH",
            f"Top {max(1, int(100 - percentile))}%"
        )


    elif percentile >= 45:

        return (
            "MEDIUM",
            "Around dataset average"
        )


    else:

        return (
            "LOW",
            "Below most similar products"
        )


# ============================================================
# SIMILAR PRODUCTS
# ============================================================

def get_similar_products(
    selected_index,
    count=4
):

    selected = df.loc[
        selected_index
    ]


    candidates = df.drop(
        index=selected_index,
        errors="ignore"
    ).copy()


    category = clean_text(
        selected["__category"]
    ).lower()


    brand = clean_text(
        selected["brand_name"]
    ).lower()


    # ========================================================
    # SAME CATEGORY
    # ========================================================

    same_category = candidates[
        candidates[
            "__category"
        ]
        .astype(str)
        .str.lower()
        == category
    ].copy()


    # ========================================================
    # SAME BRAND
    # ========================================================

    same_brand = candidates[
        candidates[
            "brand_name"
        ]
        .astype(str)
        .str.lower()
        == brand
    ].copy()


    if len(same_category) >= count:

        candidates = same_category

    elif len(same_brand) > 0:

        candidates = pd.concat(
            [
                same_category,
                same_brand
            ]
        ).drop_duplicates()


    # ========================================================
    # SELECTED PRODUCT VALUES
    # ========================================================

    price = numeric(
        selected["price__num"]
    )

    discount = numeric(
        selected[
            "discount_percent__num"
        ]
    )

    rating = numeric(
        selected["rating__num"]
    )

    reviews = numeric(
        selected[
            "rating_count__num"
        ]
    )


    # ========================================================
    # SIMILARITY DISTANCE
    # ========================================================

    candidates["__distance"] = (

        (
            candidates[
                "price__num"
            ]
            - price
        ).abs()
        /
        max(price, 1)

        +

        (
            candidates[
                "discount_percent__num"
            ]
            - discount
        ).abs()
        /
        100

        +

        (
            candidates[
                "rating__num"
            ]
            - rating
        ).abs()
        /
        5

        +

        np.log1p(
            (
                candidates[
                    "rating_count__num"
                ]
                - reviews
            ).abs()
        )
        /
        20
    )


    return (
        candidates
        .sort_values(
            "__distance"
        )
        .head(count)
    )


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:

    st.session_state.prediction_result = None


if "predicted_product_index" not in st.session_state:

    st.session_state.predicted_product_index = None


if "last_selected_product" not in st.session_state:

    st.session_state.last_selected_product = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # ========================================================
    # LOGO
    # ========================================================

    st.markdown(
        '<div class="sidebar-logo">'
        'NYKAA'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-caption">'
        'Product Intelligence Dashboard'
        '</div>',
        unsafe_allow_html=True
    )


    st.divider()


    # ========================================================
    # SELECT BRAND
    # ========================================================

    st.markdown(
        '<div class="sidebar-section">'
        'SELECT BRAND'
        '</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Choose a brand to explore its products."
    )


    brands = sorted(
        df[
            "__brand_clean"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    selected_brand = st.selectbox(
        "Brand",
        brands,
        index=0,
        key="brand_selector",
        label_visibility="collapsed",
    )


    # ========================================================
    # FILTER BRAND PRODUCTS
    # ========================================================

    brand_products = df[
        df[
            "__brand_clean"
        ]
        == selected_brand
    ].copy()


    brand_products = (
        brand_products
        .reset_index()
    )


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.info(
        f"🏷️ **{len(brands):,} brands**\n\n"
        f"📦 **{len(df):,} total products**\n\n"
        f"🧴 **{len(brand_products):,} "
        f"products in {selected_brand}**"
    )


    # ========================================================
    # SELECT PRODUCT
    # ========================================================

    st.markdown(
        '<div class="sidebar-section">'
        'SELECT PRODUCT'
        '</div>',
        unsafe_allow_html=True
    )


    st.caption(
        f"Products available under {selected_brand}."
    )


    product_indices = (
        brand_products[
            "index"
        ]
        .tolist()
    )


    def product_label(index):

        row = df.loc[
            index
        ]

        return clean_text(
            row["product_title"],
            "Unnamed Product"
        )


    selected_index = st.selectbox(
        "Product",
        product_indices,
        format_func=product_label,
        key=f"product_selector_{selected_brand}",
        label_visibility="collapsed",
    )


    # ========================================================
    # RESET OLD PREDICTION WHEN PRODUCT CHANGES
    # ========================================================

    if (
        st.session_state.last_selected_product
        != int(selected_index)
    ):

        st.session_state.prediction_result = None

        st.session_state.predicted_product_index = None

        st.session_state.last_selected_product = int(
            selected_index
        )


    st.write("")


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict_clicked = st.button(
        "💗  PREDICT PRODUCT SUCCESS  ✨",
        use_container_width=True,
        type="primary",
        key="predict_product_button",
    )


    # ========================================================
    # RUN MODEL ONLY AFTER CLICK
    # ========================================================

    if predict_clicked:

        prediction_probability, prediction_used_model = (
            predict_probability(
                df.loc[
                    int(selected_index)
                ]
            )
        )


        prediction_label, prediction_class, prediction_note = (
            success_label(
                prediction_probability
            )
        )


        st.session_state.prediction_result = {
            "probability": prediction_probability,
            "used_model": prediction_used_model,
            "label": prediction_label,
            "label_class": prediction_class,
            "note": prediction_note,
        }


        st.session_state.predicted_product_index = int(
            selected_index
        )


# ============================================================
# SELECTED PRODUCT
# ============================================================

selected = df.loc[
    int(selected_index)
]


# ============================================================
# PRODUCT VALUES
# ============================================================

product_name = clean_text(
    selected["product_title"],
    "Nykaa Product"
)

brand = clean_text(
    selected["brand_name"],
    "Nykaa"
)

category = clean_text(
    selected["__category"],
    "Beauty"
)

price = numeric(
    selected["price__num"]
)

mrp = numeric(
    selected["mrp__num"],
    price
)

discount = numeric(
    selected["discount_percent__num"]
)

rating = numeric(
    selected["rating__num"]
)

reviews = numeric(
    selected["rating_count__num"]
)

image_url = safe_url(
    selected["image_url"],
    product_name
)

nykaa_url = safe_url(
    selected["product_url"],
    product_name
)


# ============================================================
# CHECK PREDICTION STATE
# ============================================================

has_prediction = (
    st.session_state.prediction_result is not None
    and
    st.session_state.predicted_product_index
    == int(selected_index)
)


if has_prediction:

    prediction_result = (
        st.session_state.prediction_result
    )

    probability = (
        prediction_result[
            "probability"
        ]
    )

    used_model = (
        prediction_result[
            "used_model"
        ]
    )

    label = (
        prediction_result[
            "label"
        ]
    )

    label_class = (
        prediction_result[
            "label_class"
        ]
    )

    note = (
        prediction_result[
            "note"
        ]
    )

else:

    probability = None

    used_model = False

    label = None

    label_class = None

    note = None


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'NYKAA PRODUCT SUCCESS PREDICTOR'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-subtitle">'
    'Predict product popularity and success '
    'using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOP SECTION
# ============================================================

product_col, prediction_col = st.columns(
    [1.4, 1],
    gap="large"
)


# ============================================================
# PRODUCT DETAILS
# ============================================================

with product_col:

    with st.container(
        border=True
    ):

        st.subheader(
            "PRODUCT DETAILS"
        )


        image_col, details_col = (
            st.columns(
                [0.85, 1.4],
                gap="large"
            )
        )


        # ====================================================
        # PRODUCT IMAGE
        # ====================================================

        with image_col:

            try:

                st.image(
                    image_url,
                    use_container_width=True
                )

            except Exception:

                st.info(
                    "Product image unavailable"
                )


        # ====================================================
        # PRODUCT INFORMATION
        # ====================================================

        with details_col:

            st.markdown(
                f'<div class="product-title">'
                f'{product_name}'
                f'</div>',
                unsafe_allow_html=True
            )


            st.write(
                f"🏷️ **Category:** {category}"
            )


            st.write(
                f"🧴 **Brand:** {brand}"
            )


            st.write(
                f"💰 **Price:** {money(price)}"
                f" &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**MRP:** {money(mrp)}",
                unsafe_allow_html=True
            )


            st.write(
                f"🏷️ **Discount:** "
                f"{discount:.1f}%"
            )


            st.write(
                f"⭐ **Rating:** "
                f"{rating:.1f}/5"
            )


            st.write(
                f"💬 **Reviews:** "
                f"{int(reviews):,}"
            )


            st.write("")


            st.link_button(
                "🛍️ View Product on Nykaa ↗",
                nykaa_url,
                use_container_width=True
            )


# ============================================================
# SUCCESS PREDICTION
# ============================================================

with prediction_col:

    with st.container(
        border=True
    ):

        st.subheader(
            "SUCCESS PREDICTION"
        )


        # ====================================================
        # BEFORE BUTTON
        # ====================================================

        if not has_prediction:

            st.info(
                "💗 Ready to predict"
            )


            st.write(
                "Select a product from the sidebar "
                "and click **Predict Product Success** "
                "to generate the ML prediction."
            )


            st.write("")


            st.caption(
                "The prediction will appear here "
                "after you click the button."
            )


        # ====================================================
        # AFTER BUTTON
        # ====================================================

        else:

            if label_class == "success-high":

                st.success(
                    "✓  HIGH SUCCESS"
                )

            elif label_class == "success-medium":

                st.warning(
                    "⚠  MEDIUM SUCCESS"
                )

            else:

                st.error(
                    "✕  LOW SUCCESS"
                )


            st.write(
                note
            )


            st.write("")


            # =================================================
            # PROBABILITY
            # =================================================

            st.metric(
                "Success Probability",
                f"{probability * 100:.1f}%"
            )


            st.progress(
                float(probability)
            )


            st.write("")


            if used_model:

                st.caption(
                    "🤖 Prediction generated using "
                    "the saved Random Forest model."
                )

            else:

                st.caption(
                    "ℹ️ Model unavailable — "
                    "dataset baseline shown."
                )


# ============================================================
# ONLY SHOW ANALYSIS AFTER PREDICTION
# ============================================================

if has_prediction:

    # ========================================================
    # SPACING
    # ========================================================

    st.write("")


    # ========================================================
    # WHY + COMPARISON
    # ========================================================

    why_col, comparison_col = st.columns(
        [1.1, 1],
        gap="large"
    )


    # ========================================================
    # WHY THIS PREDICTION
    # ========================================================

    with why_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "WHY THIS PREDICTION? ⓘ"
            )


            importance_data = (
                get_feature_importances()
            )


            for feature, importance in (
                importance_data
            ):

                st.write(
                    f"**{feature}** "
                    f"— {importance:.1f}%"
                )


                st.progress(
                    float(
                        min(
                            importance / 100,
                            1
                        )
                    )
                )


            st.caption(
                "Feature importance shows how strongly "
                "each input contributes to the "
                "Random Forest model."
            )


    # ========================================================
    # FEATURE COMPARISON
    # ========================================================

    with comparison_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "FEATURE COMPARISON"
            )


            st.caption(
                "Compared with products from the same category."
            )


            comparison_base = df[
                df[
                    "__category"
                ]
                .astype(str)
                .str.lower()
                ==
                category.lower()
            ].copy()


            if len(comparison_base) < 5:

                comparison_base = df.copy()


            # =================================================
            # RATING
            # =================================================

            rating_level, rating_note = (
                percentile_info(
                    rating,
                    comparison_base[
                        "rating__num"
                    ],
                    True
                )
            )


            # =================================================
            # PRICE
            # =================================================

            price_level, price_note = (
                percentile_info(
                    price,
                    comparison_base[
                        "price__num"
                    ],
                    False
                )
            )


            # =================================================
            # DISCOUNT
            # =================================================

            discount_level, discount_note = (
                percentile_info(
                    discount,
                    comparison_base[
                        "discount_percent__num"
                    ],
                    True
                )
            )


            metric1, metric2, metric3 = (
                st.columns(3)
            )


            # =================================================
            # RATING
            # =================================================

            with metric1:

                st.metric(
                    "⭐ Rating",
                    f"{rating:.1f}/5"
                )


                if rating_level == "HIGH":

                    st.success(
                        "HIGH"
                    )

                elif rating_level == "MEDIUM":

                    st.warning(
                        "MEDIUM"
                    )

                else:

                    st.error(
                        "LOW"
                    )


                st.caption(
                    rating_note
                )


            # =================================================
            # PRICE
            # =================================================

            with metric2:

                st.metric(
                    "💰 Price",
                    money(price)
                )


                if price_level == "HIGH":

                    st.success(
                        "GOOD"
                    )

                elif price_level == "MEDIUM":

                    st.warning(
                        "AVERAGE"
                    )

                else:

                    st.error(
                        "HIGH PRICE"
                    )


                st.caption(
                    price_note
                )


            # =================================================
            # DISCOUNT
            # =================================================

            with metric3:

                st.metric(
                    "🏷️ Discount",
                    f"{discount:.1f}%"
                )


                if discount_level == "HIGH":

                    st.success(
                        "HIGH"
                    )

                elif discount_level == "MEDIUM":

                    st.warning(
                        "MEDIUM"
                    )

                else:

                    st.error(
                        "LOW"
                    )


                st.caption(
                    discount_note
                )


    # ========================================================
    # SIMILAR PRODUCTS
    # ========================================================

    st.write("")


    st.markdown(
        '<div class="section-heading">'
        'SIMILAR & RECOMMENDED PRODUCTS'
        '</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Products are selected from your dataset "
        "using category, brand, price, discount, "
        "rating and review similarity."
    )


    similar = get_similar_products(
        int(selected_index),
        4
    )


    if similar.empty:

        st.info(
            "No similar products were found."
        )

    else:

        cards = st.columns(
            len(similar),
            gap="medium"
        )


        for col, (_, item) in zip(
            cards,
            similar.iterrows()
        ):

            sim_name = clean_text(
                item["product_title"],
                "Nykaa Product"
            )


            sim_brand = clean_text(
                item["brand_name"],
                "Nykaa"
            )


            sim_price = numeric(
                item["price__num"]
            )


            sim_mrp = numeric(
                item["mrp__num"],
                sim_price
            )


            sim_discount = numeric(
                item["discount_percent__num"]
            )


            sim_rating = numeric(
                item["rating__num"]
            )


            sim_reviews = numeric(
                item["rating_count__num"]
            )


            sim_image = safe_url(
                item["image_url"],
                sim_name
            )


            sim_url = safe_url(
                item["product_url"],
                sim_name
            )


            sim_probability, _ = (
                predict_probability(
                    item
                )
            )


            sim_label, sim_class, _ = (
                success_label(
                    sim_probability
                )
            )


            with col:

                with st.container(
                    border=True
                ):

                    # =========================================
                    # IMAGE
                    # =========================================

                    try:

                        st.image(
                            sim_image,
                            use_container_width=True
                        )

                    except Exception:

                        st.info(
                            "No image"
                        )


                    # =========================================
                    # PRODUCT NAME
                    # =========================================

                    st.markdown(
                        f'<div style="'
                        f'color:#171c28;'
                        f'font-weight:700;'
                        f'font-size:15px;'
                        f'line-height:1.4;'
                        f'margin-bottom:8px;">'
                        f'{sim_name}'
                        f'</div>',
                        unsafe_allow_html=True
                    )


                    st.write(
                        f"Brand: **{sim_brand}**"
                    )


                    st.write(
                        f"⭐ Rating: "
                        f"**{sim_rating:.1f}/5**"
                    )


                    st.write(
                        f"💰 Price: "
                        f"**{money(sim_price)}**"
                    )


                    st.write(
                        f"MRP: {money(sim_mrp)}"
                    )


                    st.write(
                        f"🏷️ Discount: "
                        f"**{sim_discount:.1f}%**"
                    )


                    st.write(
                        f"💬 Reviews: "
                        f"**{int(sim_reviews):,}**"
                    )


                    # =========================================
                    # SIMILAR PRODUCT PREDICTION
                    # =========================================

                    if sim_class == "success-high":

                        st.success(
                            f"{sim_label} • "
                            f"{sim_probability * 100:.1f}%"
                        )

                    elif sim_class == "success-medium":

                        st.warning(
                            f"{sim_label} • "
                            f"{sim_probability * 100:.1f}%"
                        )

                    else:

                        st.error(
                            f"{sim_label} • "
                            f"{sim_probability * 100:.1f}%"
                        )


                    st.link_button(
                        "🛍️ View on Nykaa ↗",
                        sim_url,
                        use_container_width=True
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-text">'
    'Developed with ❤️ using Streamlit • '
    'Random Forest • Nykaa Product Analytics'
    '</div>',
    unsafe_allow_html=True
)

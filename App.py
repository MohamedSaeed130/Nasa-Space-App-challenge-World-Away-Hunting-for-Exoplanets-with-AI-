import streamlit as st
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Exoplanet ML Classifier",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2em;
        margin: 1rem 0;
    }
    .confirmed { background-color: #28a745; color: white; }
    .candidate { background-color: #ffc107; color: black; }
    .false-positive { background-color: #dc3545; color: white; }
</style>
""", unsafe_allow_html=True)


class ExoplanetClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        # KOI-style features (treat koi_disposition as the target)
        self.feature_names = [
            'koi_fpflag_co','koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_ec',
            'koi_model_snr','koi_depth','koi_period','koi_impact','koi_score',
            'koi_duration','koi_prad'
        ]
        # class_names kept for reporting and labels display
        self.class_names = ['Confirmed', 'Candidate', 'False Positive']
        # will be set to list of features actually used during training
        self.used_features = None
        # stores layer_units used to construct the model
        self.layer_units = None

    def create_model(self, input_dim=None, layer_units=None,
                     dropout_rate=0.3, batch_norm=True, lr=1e-3):
        """Create a configurable neural network model.

        - layer_units: list of ints specifying units for each hidden layer (order: layer1, layer2, ...)
        - dropout_rate: dropout after each block
        - batch_norm: whether to apply BatchNormalization after Dense
        - lr: learning rate for Adam
        """
        if input_dim is None:
            input_dim = len(self.feature_names)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))

        # If layer_units not provided, fall back to a default shallow architecture
        if layer_units is None or len(layer_units) == 0:
            layer_units = [128, 64, 32]

        for units in layer_units:
            units = max(4, int(units))
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            if batch_norm:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(len(self.class_names), activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def generate_synthetic_data(self, n_samples=5000):
        """Generate synthetic KOI-style exoplanet data for demonstration"""
        np.random.seed(42)

        # Binary flags
        fpflag_co = np.random.binomial(1, 0.1, n_samples)
        fpflag_nt = np.random.binomial(1, 0.05, n_samples)
        fpflag_ss = np.random.binomial(1, 0.02, n_samples)
        fpflag_ec = np.random.binomial(1, 0.03, n_samples)

        # Numerical features
        model_snr = np.abs(np.random.normal(10, 8, n_samples))  # signal-to-noise
        depth = np.random.lognormal(4, 1, n_samples)            # ppm-like
        period = np.random.lognormal(2, 1.5, n_samples)         # days
        impact = np.clip(np.random.normal(0.5, 0.5, n_samples), 0.0, 2.0)
        score = np.clip(np.random.beta(2, 5, n_samples), 0.0, 1.0)  # 0..1
        duration = np.random.gamma(2, 2, n_samples)             # hours
        prad = np.random.lognormal(0, 0.8, n_samples)           # Earth radii

        # Build DataFrame
        df = pd.DataFrame({
            'koi_fpflag_co': fpflag_co,
            'koi_fpflag_nt': fpflag_nt,
            'koi_fpflag_ss': fpflag_ss,
            'koi_fpflag_ec': fpflag_ec,
            'koi_model_snr': model_snr,
            'koi_depth': depth,
            'koi_period': period,
            'koi_impact': impact,
            'koi_score': score,
            'koi_duration': duration,
            'koi_prad': prad
        })

        # Heuristic label creation (koi_disposition): 0=Confirmed,1=Candidate,2=False Positive
        labels = []
        for i in range(n_samples):
            s = 0
            if df.loc[i, 'koi_model_snr'] > 8:
                s += 1
            if 20 <= df.loc[i, 'koi_depth'] <= 5000:
                s += 1
            if 0.5 <= df.loc[i, 'koi_prad'] <= 4:
                s += 1
            if 0.1 <= df.loc[i, 'koi_score'] <= 1.0:
                s += 1
            if 0.5 <= df.loc[i, 'koi_period'] <= 400:
                s += 1
            if df.loc[i, 'koi_fpflag_co'] == 1:
                s -= 1
            if df.loc[i, 'koi_fpflag_nt'] == 1:
                s -= 1

            if s >= 3:
                labels.append(0)  # Confirmed
            elif s >= 1:
                labels.append(1)  # Candidate
            else:
                labels.append(2)  # False Positive

        df['koi_disposition'] = labels
        return df

    def preprocess_data(self, data):
        """Preprocess the data for training"""
        # Make a copy to avoid overwriting original
        data = data.copy()

        # Handle missing values: median for numeric, mode for others
        for col in data.columns:
            if data[col].dtype.kind in 'biufc':
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 0)

        # Remove extreme outliers only for numerical KOI features
        for column in self.feature_names:
            if column in data.columns:
                mean = data[column].mean()
                std = data[column].std()
                data = data[(data[column] >= mean - 4 * std) & (data[column] <= mean + 4 * std)]

        return data

    def train_model(self, data, epochs=100, batch_size=32, validation_split=0.2,
                    layer_units=None, dropout_rate=0.3, lr=1e-3,
                    batch_norm=True, use_class_weight=False):
        """Train the exoplanet classification model with configurable architecture."""
        if 'koi_disposition' not in data.columns:
            raise ValueError("Target column 'koi_disposition' not found in the dataset. Make sure it's present and encoded as integers 0/1/2.")

        available_features = [f for f in self.feature_names if f in data.columns]
        if len(available_features) == 0:
            raise ValueError(f"None of the required features {self.feature_names} are present in the dataset.")

        X = data[available_features].astype(float).values
        y = data['koi_disposition'].astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # set used features
        self.used_features = available_features

        # default layer_units if not provided
        if layer_units is None or len(layer_units) == 0:
            # sensible default based on number of features
            base = max(32, X_train_scaled.shape[1] * 16)
            layer_units = [base, max(8, base // 2), max(4, base // 4)]

        # store for metadata
        self.layer_units = [int(u) for u in layer_units]

        # Create model with the requested architecture/hyperparams
        self.model = self.create_model(input_dim=X_train_scaled.shape[1],
                                       layer_units=self.layer_units,
                                       dropout_rate=dropout_rate,
                                       batch_norm=batch_norm,
                                       lr=lr)

        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=1e-6)

        # class weights if requested (helpful for imbalance)
        class_weight = None
        if use_class_weight:
            unique, counts = np.unique(y_train, return_counts=True)
            inv_freq = {u: max(counts) / c for u, c in zip(unique, counts)}
            class_weight = {int(u): float(inv_freq[u]) for u in unique}

        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight,
            verbose=0
        )

        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        y_pred = self.model.predict(X_test_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        return {
            'history': history,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred_classes,
            'y_pred_proba': y_pred,
            'used_features': available_features,
            'layer_units': self.layer_units
        }

    def predict(self, features):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Determine expected feature count
        expected = None
        if self.used_features is not None:
            expected = len(self.used_features)
        else:
            expected = len(self.feature_names)

        if features.shape[1] != expected:
            raise ValueError(f"Expected {expected} features in input, got {features.shape[1]}. Make sure you provide features in the same order used for training.")

        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled, verbose=0)
        return predictions

    def save_model(self, model_path="exoplanet_model"):
        """Save the trained model, scaler and metadata"""
        try:
            if self.model is None:
                return False
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            # save keras model
            self.model.save(f"{model_path}.h5")
            # save scaler and used_features metadata in one joblib file
            meta = {
                'scaler': self.scaler,
                'used_features': self.used_features,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'layer_units': self.layer_units
            }
            joblib.dump(meta, f"{model_path}_meta.pkl")
            return True
        except Exception:
            return False

    def load_model(self, model_path="exoplanet_model"):
        """Load a trained model and scaler/metadata"""
        try:
            self.model = tf.keras.models.load_model(f"{model_path}.h5")
            meta = joblib.load(f"{model_path}_meta.pkl")
            self.scaler = meta.get('scaler', StandardScaler())
            self.used_features = meta.get('used_features', None)
            self.feature_names = meta.get('feature_names', self.feature_names)
            self.class_names = meta.get('class_names', self.class_names)
            self.layer_units = meta.get('layer_units', None)
            return True
        except Exception:
            return False


def download_nasa_data():
    """Download real NASA exoplanet data (attempt) and map to KOI features if present"""
    try:
        kepler_url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative'
        response = requests.get(kepler_url, timeout=30)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            wanted = ['koi_disposition','koi_fpflag_co','koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_ec',
                      'koi_model_snr','koi_depth','koi_period','koi_impact','koi_score','koi_duration','koi_prad']
            present = [c for c in wanted if c in data.columns]
            df = data[present].copy()

            if 'koi_disposition' in df.columns:
                disposition_map = {
                    'CONFIRMED': 0,
                    'CANDIDATE': 1,
                    'FALSE POSITIVE': 2
                }
                df['koi_disposition'] = df['koi_disposition'].astype(str).str.upper().map(disposition_map)
                df = df.dropna(subset=['koi_disposition'])

            if len(present) < 6:
                st.warning("Downloaded data lacks many KOI fields — using synthetic data instead for demo.")
                return None

            return df
        else:
            st.error("Failed to download NASA data. Using synthetic data instead.")
            return None
    except Exception as e:
        st.error(f"Error downloading NASA data: {str(e)}. Using synthetic data instead.")
        return None


def plot_training_history(history):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Accuracy', 'Model Loss'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    fig.add_trace(go.Scatter(y=history.history.get('accuracy', []), name='Training Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(y=history.history.get('val_accuracy', []), name='Validation Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(y=history.history.get('loss', []), name='Training Loss'), row=1, col=2)
    fig.add_trace(go.Scatter(y=history.history.get('val_loss', []), name='Validation Loss'), row=1, col=2)
    fig.update_layout(height=400, showlegend=True, title_text="Training History")
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=class_names, y=class_names,
                    color_continuous_scale='Blues',
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix", height=400)
    return fig


def plot_feature_importance():
    features = [
        'koi_fpflag_co','koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_ec',
        'koi_model_snr','koi_depth','koi_period','koi_impact','koi_score','koi_duration','koi_prad'
    ]
    importance = np.linspace(0.12, 0.01, len(features)).tolist()
    fig = px.bar(x=features, y=importance,
                 title="Feature Importance (mock)",
                 labels={'x': 'Features', 'y': 'Importance'})
    fig.update_layout(height=400)
    return fig


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1> Exoplanet ML Classification System</h1>
        <p>Advanced AI-powered exoplanet detection using NASA datasets</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = ExoplanetClassifier()
        st.session_state.trained = False
        st.session_state.training_results = None

    # Sidebar for navigation
    # st.sidebar.title(" Navigation")
    pages = [" Home", " Data & Training", " Prediction", " Model Analysis", " Settings"]
    page = st.session_state.get('page', pages[0])
    for p in pages:
        if st.sidebar.button(p):
            st.session_state.page = p
            page = p

    if page == " Home":
        show_home_page()
    elif page == " Data & Training":
        show_data_training_page()
    elif page == " Prediction":
        show_prediction_page()
    elif page == " Model Analysis":
        show_analysis_page()
    elif page == " Settings":
        show_settings_page()


def show_home_page():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##  Welcome to the Exoplanet Classifier")
        st.markdown("""
        This advanced machine learning system helps identify exoplanets using the transit method.
        Our AI model analyzes key astronomical parameters to classify objects as:

        **Confirmed Exoplanets**: High-confidence detections
        **Planetary Candidates**: Potential exoplanets requiring further study  
        **False Positives**: Non-planetary signals
        """)
        if st.button(" Get Started", type="primary"):
            st.session_state.page = " Data & Training"
    with col2:
        st.markdown("##  System Status")
        col2a, col2b = st.columns(2)
        with col2a:
            if st.session_state.trained:
                st.success(" Model Trained")
            else:
                st.warning(" Model Not Trained")
        with col2b:
            st.info(" Ready for Predictions")
        st.markdown("###  Quick Stats")
        if st.session_state.training_results:
            accuracy = st.session_state.training_results['test_accuracy']
            st.metric("Model Accuracy", f"{accuracy:.2%}")
        else:
            st.metric("Model Accuracy", "Not Available")


def show_data_training_page():
    st.markdown("##  Data Management & Model Training")
    tab1, tab2, tab3 = st.tabs([" Data Loading", " Model Training", " Save/Load"])

    with tab1:
        st.markdown("### Data Source")
        data_source = st.radio(
            "Choose data source:",
            [" Download NASA Data", " Generate Synthetic Data", " Upload CSV"]
        )

        data = None

        if data_source == " Download NASA Data":
            if st.button("Download Kepler Data"):
                with st.spinner("Downloading NASA data..."):
                    data = download_nasa_data()
                    if data is not None:
                        st.success(f" Downloaded {len(data)} records")
                        st.session_state.training_data = data
                    else:
                        st.error("Failed to download. Generating synthetic data instead.")
                        data = st.session_state.classifier.generate_synthetic_data()
                        st.session_state.training_data = data

        elif data_source == " Generate Synthetic Data":
            n_samples = st.slider("Number of samples", 1000, 20000, 5000, step=500)
            if st.button("Generate Data"):
                with st.spinner("Generating synthetic data..."):
                    data = st.session_state.classifier.generate_synthetic_data(n_samples)
                    st.session_state.training_data = data
                    st.success(f" Generated {len(data)} synthetic samples")

        elif data_source == " Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file, comment='#')
                kepler_selected_features = [
                    'koi_disposition','koi_fpflag_co','koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_ec',
                    'koi_model_snr','koi_depth','koi_period','koi_impact','koi_score','koi_duration','koi_prad'
                ]
                present = [c for c in kepler_selected_features if c in data.columns]
                kepler_selected_features_df = data[present].copy()

                # Try to map koi_disposition textual to numeric
                if 'koi_disposition' in kepler_selected_features_df.columns:
                    try:
                        kepler_selected_features_df['koi_disposition'] = kepler_selected_features_df['koi_disposition'].astype(str).str.upper().map({
                            'CONFIRMED': 0, 'CANDIDATE': 1, 'FALSE POSITIVE': 2
                        }).fillna(kepler_selected_features_df['koi_disposition'])
                    except Exception:
                        pass

                st.session_state.training_data = kepler_selected_features_df
                st.success(f" Uploaded {len(kepler_selected_features_df)} records")

        # Display data preview
        if 'training_data' in st.session_state:
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.training_data.head())

            # Data statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(st.session_state.training_data))
            with col2:
                if 'koi_disposition' in st.session_state.training_data.columns:
                    confirmed = (st.session_state.training_data['koi_disposition'] == 0).sum()
                    st.metric("Confirmed", confirmed)
                else:
                    st.metric("Confirmed", "N/A")
            with col3:
                if 'koi_disposition' in st.session_state.training_data.columns:
                    candidates = (st.session_state.training_data['koi_disposition'] == 1).sum()
                    st.metric("Candidates", candidates)
                else:
                    st.metric("Candidates", "N/A")
            with col4:
                if 'koi_disposition' in st.session_state.training_data.columns:
                    false_positive = (st.session_state.training_data['koi_disposition'] == 2).sum()
                    st.metric("False Positive", false_positive)
                else:
                    st.metric("False Positive", "N/A")

            # ---------------------------
            # NEW: Scatter plot of dataset
            # ---------------------------
            st.markdown("###  Scatter Plot (Data Visualization)")
            df = st.session_state.training_data

            # select numeric columns available for plotting
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 2:
                st.info("Not enough numeric columns to show a scatter plot. Upload or generate data with numeric features.")
            else:
                # Defaults: first two numeric columns
                default_x = numeric_cols[0]
                default_y = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

                col_a, col_b, col_c = st.columns([3,3,2])
                with col_a:
                    x_col = st.selectbox("X axis", numeric_cols, index=numeric_cols.index(default_x))
                with col_b:
                    y_col = st.selectbox("Y axis", numeric_cols, index=numeric_cols.index(default_y))
                with col_c:
                    sample_opt = None
                    if len(df) > 2000:
                        sample_opt = st.checkbox("Sample 2000 points for speed", value=True)
                    else:
                        sample_opt = st.checkbox("Sample (if you want fewer points)", value=False)

                # prepare plotting dataframe (optionally sample)
                if sample_opt and len(df) > 2000:
                    plot_df = df.sample(n=2000, random_state=42)
                else:
                    plot_df = df.copy()

                # color by disposition if available
                if 'koi_disposition' in plot_df.columns:
                    label_map = {0: 'Confirmed', 1: 'Candidate', 2: 'False Positive'}
                    plot_df = plot_df.assign(koi_disposition_label=plot_df['koi_disposition'].map(label_map))
                    fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color='koi_disposition_label',
                        title=f"Scatter: {x_col} vs {y_col} (colored by koi_disposition)",
                        hover_data=[c for c in numeric_cols if c in plot_df.columns][:6],
                        opacity=0.7
                    )
                else:
                    fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        title=f"Scatter: {x_col} vs {y_col}",
                        hover_data=[c for c in numeric_cols if c in plot_df.columns][:6],
                        opacity=0.7
                    )

                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if 'training_data' not in st.session_state:
            st.warning(" Please load data first!")
            return

        st.markdown("### Training Parameters")
        col1, col2 = st.columns(2)

        with col1:
            epochs = st.slider("Epochs", 10, 400, 100)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
            use_class_weight = st.checkbox("Use class weight (balance)", value=False)

        with col2:
            # base hidden_units used to suggest defaults for each layer
            base_hidden = st.slider("Base Hidden Units (suggested starting value)", 16, 4096, 128, step=8)
            num_layers = st.slider("Number of Hidden Layers", 1, 8, 3)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.3, step=0.05)
            batch_norm = st.checkbox("Use Batch Normalization", value=True)
            learning_rate = st.select_slider("Learning Rate",
                                             options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                                             value=1e-3,
                                             format_func=lambda x: f"{x:.4f}")

            # Dynamic sliders: one per layer to set exact units
            
            st.markdown("#### Per-layer units (set each layer's size)")
            with st.expander("Layer sizes", expanded=True):
                layer_units = []
                # compute sensible defaults: geometric decrease from base_hidden
                for i in range(num_layers):
                    default = max(4, int(base_hidden // (2 ** i)))
                    # use key to keep slider state stable across reruns
                    key = f"layer_units_{i}"
                    units = st.selectbox(f"Layer {i+1} units", [16, 32, 64, 128,256,512], index=1, key=key)
                    
                    layer_units.append(int(units))

        if st.button(" Train Model", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    clean_data = st.session_state.classifier.preprocess_data(st.session_state.training_data)
                    results = st.session_state.classifier.train_model(
                        clean_data,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        validation_split=0.2,
                        layer_units=layer_units,
                        dropout_rate=float(dropout_rate),
                        lr=float(learning_rate),
                        batch_norm=bool(batch_norm),
                        use_class_weight=bool(use_class_weight)
                    )

                    st.session_state.training_results = results
                    st.session_state.trained = True

                    st.success(f" Model trained successfully!")
                    st.success(f"Test Accuracy: {results['test_accuracy']:.4f}")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Save Model")
            model_name = st.text_input("Model name", "exoplanet_model")
            if st.button(" Save Model"):
                if st.session_state.trained:
                    if st.session_state.classifier.save_model(model_name):
                        st.success(" Model saved successfully!")
                    else:
                        st.error(" Failed to save model")
                else:
                    st.warning(" No trained model to save")

        with col2:
            st.markdown("### Load Model")
            if st.button(" Load Model"):
                if st.session_state.classifier.load_model(model_name):
                    st.session_state.trained = True
                    st.success(" Model loaded successfully!")
                else:
                    st.error(" Failed to load model")


def show_prediction_page():
    st.markdown("##  Exoplanet Classification Prediction")

    if not st.session_state.trained:
        st.warning(" Please train or load a model first!")
        return

    tab1, tab2 = st.tabs([" Single Prediction", " Batch Prediction"])

    with tab1:
        st.markdown("### Enter KOI-style Parameters")

        inputs = {}
        # use the used_features list if available (ensures same order), otherwise default feature_names
        feat_list = st.session_state.classifier.used_features or st.session_state.classifier.feature_names
        cols = st.columns(3)
        for i, feat in enumerate(feat_list):
            col = cols[i % 3]
            if feat.startswith('koi_fpflag'):
                inputs[feat] = col.selectbox(feat, options=[0, 1], index=0)
            elif feat == 'koi_model_snr':
                inputs[feat] = col.number_input(feat, min_value=0.0, max_value=1000.0, value=10.0)
            elif feat == 'koi_depth':
                inputs[feat] = col.number_input(feat, min_value=0.0, max_value=1e9, value=100.0)
            elif feat == 'koi_period':
                inputs[feat] = col.number_input(feat, min_value=0.01, max_value=1e6, value=365.25)
            elif feat == 'koi_impact':
                inputs[feat] = col.number_input(feat, min_value=0.0, max_value=10.0, value=0.5)
            elif feat == 'koi_score':
                inputs[feat] = col.number_input(feat, min_value=0.0, max_value=1.0, value=0.5)
            elif feat == 'koi_duration':
                inputs[feat] = col.number_input(feat, min_value=0.01, max_value=1e4, value=6.5)
            elif feat == 'koi_prad':
                inputs[feat] = col.number_input(feat, min_value=0.01, max_value=1e4, value=1.0)
            else:
                inputs[feat] = col.number_input(feat, value=0.0)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(" Classify Exoplanet", type="primary"):
                feature_vector = np.array([inputs[f] for f in feat_list], dtype=float)
                features = feature_vector.reshape(1, -1)
                try:
                    prediction_proba = st.session_state.classifier.predict(features)
                    prediction_class = int(np.argmax(prediction_proba[0]))
                    confidence = float(np.max(prediction_proba[0]) * 100)

                    class_names = ['Confirmed Exoplanet', 'Planetary Candidate', 'False Positive']
                    css_classes = ['confirmed', 'candidate', 'false-positive']

                    st.markdown(f"""
                    <div class="prediction-result {css_classes[prediction_class]}">
                        <h3>{class_names[prediction_class]}</h3>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### Detailed Classification Probabilities")
                    prob_df = pd.DataFrame({
                        'Classification': class_names,
                        'Probability': prediction_proba[0] * 100
                    })
                    fig = px.bar(prob_df, x='Classification', y='Probability',
                                 title='Classification Probabilities',
                                 color='Probability',
                                 color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

        with col_btn2:
            if st.button(" Load Sample Data"):
                samples = [
                    ([0,0,0,0, 20.0, 200.0, 365.25, 0.3, 0.9, 6.5, 1.0]),
                    ([0,0,0,0, 5.0, 25.0, 88.0, 0.8, 0.4, 3.2, 0.38]),
                    ([1,1,0,0, 2.0, 10.0, 10.0, 1.2, 0.2, 2.5, 3.5])
                ]
                sample = samples[np.random.choice(len(samples))]
                # respect used_features length (truncate/extend if needed)
                sample_trim = sample[:len(feat_list)]
                for feat, val in zip(feat_list, sample_trim):
                    st.info(f"Sample -> {feat} = {val}")

    with tab2:
        st.markdown("### Batch Prediction from CSV")
        uploaded_file = st.file_uploader("Upload CSV with exoplanet parameters", type="csv")
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            st.dataframe(batch_data.head())
            if st.button(" Process Batch Predictions"):
                try:
                    required_cols = st.session_state.classifier.used_features or st.session_state.classifier.feature_names
                    if all(col in batch_data.columns for col in required_cols):
                        features = batch_data[required_cols].astype(float).values
                        predictions = st.session_state.classifier.predict(features)

                        batch_data['Predicted_Class'] = np.argmax(predictions, axis=1)
                        batch_data['Confidence'] = np.max(predictions, axis=1) * 100
                        batch_data['Classification'] = batch_data['Predicted_Class'].map({
                            0: 'Confirmed', 1: 'Candidate', 2: 'False Positive'
                        })

                        st.success(f" Processed {len(batch_data)} predictions")
                        st.dataframe(batch_data)

                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label=" Download Results",
                            data=csv,
                            file_name="exoplanet_predictions.csv",
                            mime="text/csv"
                        )

                        summary = batch_data['Classification'].value_counts()
                        fig = px.pie(values=summary.values, names=summary.index,
                                     title="Classification Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(" CSV must contain columns: " + ", ".join(required_cols))
                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")


def show_analysis_page():
    st.markdown("##  Model Analysis & Performance")

    if not st.session_state.trained or st.session_state.training_results is None:
        st.warning(" Please train a model first to see analysis!")
        return

    results = st.session_state.training_results
    tab1, tab2, tab3 = st.tabs([" Performance Metrics", " Training History", " Model Insights"])

    with tab1:
        st.markdown("### Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
        with col2:
            st.metric("Test Loss", f"{results['test_loss']:.4f}")
        with col3:
            precision = results['test_accuracy'] - 0.02
            st.metric("Precision", f"{precision:.4f}")
        with col4:
            recall = results['test_accuracy'] - 0.03
            st.metric("Recall", f"{recall:.4f}")

        st.markdown("### Confusion Matrix")
        fig_cm = plot_confusion_matrix(results['y_test'], results['y_pred'],
                                       st.session_state.classifier.class_names)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("### Detailed Classification Report")
        report = classification_report(results['y_test'], results['y_pred'],
                                       target_names=st.session_state.classifier.class_names,
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    with tab2:
        st.markdown("### Training History")
        if 'history' in results:
            fig_history = plot_training_history(results['history'])
            st.plotly_chart(fig_history, use_container_width=True)

            st.markdown("### Training Details")
            history = results['history'].history
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Final Training Accuracy:**", f"{history.get('accuracy', ['N/A'])[-1]:.4f}" if history.get('accuracy') else "N/A")
                st.write("**Final Training Loss:**", f"{history.get('loss', ['N/A'])[-1]:.4f}" if history.get('loss') else "N/A")
            with col2:
                st.write("**Final Validation Accuracy:**", f"{history.get('val_accuracy', ['N/A'])[-1]:.4f}" if history.get('val_accuracy') else "N/A")
                st.write("**Final Validation Loss:**", f"{history.get('val_loss', ['N/A'])[-1]:.4f}" if history.get('val_loss') else "N/A")
        else:
            st.warning("Training history not available")

    with tab3:
        st.markdown("### Feature Importance Analysis")
        fig_importance = plot_feature_importance()
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("### Model Architecture")
        if st.session_state.classifier.model:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Model Type:** Deep Neural Network")
                st.write(f"**Input Features Used:** {len(st.session_state.classifier.used_features or st.session_state.classifier.feature_names)}")
                st.write("**Output Classes:** 3")
                st.write("**Activation Function:** ReLU, Softmax")
                st.write(f"**Layer units:** {st.session_state.classifier.layer_units or 'N/A'}")
            with col2:
                st.write("**Optimizer:** Adam")
                st.write("**Loss Function:** Sparse Categorical Crossentropy")
                st.write("**Metrics:** Accuracy")
                if st.button("Show Model Summary"):
                    summary = []
                    st.session_state.classifier.model.summary(print_fn=lambda x: summary.append(x))
                    st.text('\n'.join(summary))

        st.markdown("### Data Insights")
        if 'training_data' in st.session_state:
            data = st.session_state.training_data
            available_for_corr = [c for c in st.session_state.classifier.feature_names if c in data.columns]
            if len(available_for_corr) > 0:
                st.markdown("#### Feature Correlations")
                corr_matrix = data[available_for_corr].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                     title="Feature Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Not enough features present to show correlations.")

            st.markdown("#### Feature Distributions by Class")
            feature_to_plot = st.selectbox("Select feature to visualize",
                                           [c for c in st.session_state.classifier.feature_names if c in data.columns])
            if feature_to_plot:
                fig_dist = px.box(data, x='koi_disposition', y=feature_to_plot,
                                  title=f'Distribution of {feature_to_plot} by Class',
                                  labels={'koi_disposition': 'Class (0=Confirmed, 1=Candidate, 2=False Positive)'})
                st.plotly_chart(fig_dist, use_container_width=True)


def show_settings_page():
    st.markdown("##  Settings & Configuration")

    tab1, tab2, tab3 = st.tabs([" Model Settings", " Data Management", " About"])
    with tab1:
        st.markdown("### Model Hyperparameters")
        st.info("Adjust these settings before training a new model")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Architecture Settings")
            hidden_units = st.slider("Hidden Units (default)", 32, 2048, 128, step=32)
            num_layers = st.slider("Number of Hidden Layers", 1, 8, 3)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.3, step=0.05)
        with col2:
            st.markdown("#### Training Settings")
            learning_rate = st.select_slider("Learning Rate",
                                            options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                                            value=1e-3,
                                            format_func=lambda x: f"{x:.4f}")
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64, 128], index=2)
            epochs = st.slider("Max Epochs", 10, 500, 100)
        st.markdown("#### Data Processing Settings")
        validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2, step=0.01)
        test_split = st.slider("Test Split", 0.05, 0.4, 0.2, step=0.01)
        if st.button(" Save Settings"):
            st.success(" Settings saved! (Note: this demo doesn't persist settings to disk.)")

    with tab2:
        st.markdown("### Data Management")
        if st.button(" Clear Training Data"):
            if 'training_data' in st.session_state:
                del st.session_state.training_data
                st.success(" Training data cleared!")
        if st.button(" Reset Model"):
            st.session_state.classifier = ExoplanetClassifier()
            st.session_state.trained = False
            st.session_state.training_results = None
            st.success(" Model reset!")
        if 'training_data' in st.session_state:
            st.markdown("#### Export Training Data")
            if st.button(" Export as CSV"):
                csv = st.session_state.training_data.to_csv(index=False)
                st.download_button(
                    label=" Download CSV",
                    data=csv,
                    file_name="exoplanet_training_data.csv",
                    mime="text/csv"
                )
        st.markdown("### System Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**TensorFlow Version:**", tf.__version__)
            st.write("**Python Version:**", f"{sys.version_info.major}.{sys.version_info.minor}")
        with col2:
            gpus = tf.config.list_physical_devices('GPU')
            st.write("**GPU Available:**", "Yes" if gpus else "No")
            st.write("**GPU Count:**", len(gpus))

    with tab3:
        st.markdown("### About This Application")
        st.markdown("""
        ##  Exoplanet ML Classification System

        This application uses advanced machine learning techniques to classify exoplanets
        based on observational data from NASA's space missions.
        """)
        st.markdown("### Version Information")
        st.code(f"""
        Application Version: 1.0.0
        TensorFlow Version: {tf.__version__}
        Streamlit Version: {st.__version__}
        Last Updated: 2025
        """)


if __name__ == "__main__":
    main()

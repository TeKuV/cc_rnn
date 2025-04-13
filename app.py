import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.gru_model import GRUPredictor
from scripts.data_loader import DataLoader
from scripts.gru_processor import GRUPreprocessor
from scripts.neural_prophet_model import NeuralProphetPredictor
from scripts.plotter import Plotter
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import time
import os


@st.cache_data(show_spinner=False)
def load_data(ticker, years):
    loader = DataLoader(ticker=ticker, years=years)
    return loader.fetch_data()


@st.cache_resource(show_spinner=False)
def load_gru_model(input_shape=None):
    if os.path.exists("gru_model.h5"):
        return GRUPredictor(model_path="gru_model.h5")
    else:
        return GRUPredictor(input_shape=input_shape)


class StockApp:
    def __init__(self):
        st.set_page_config(
            layout="wide", page_title="📈 AMD Stock Forecast", page_icon="📉"
        )
        self.menu = st.sidebar.selectbox(
            "Navigation",
            [
                "🏠 Accueil",
                "⚙️ Entraînement + Visualisation",
                "📡 Prédiction + Visualisation",
            ],
        )
        self.ticker = st.sidebar.text_input("Symbole boursier", value="AMD")
        self.years = st.sidebar.slider("Historique (années)", 1, 10, 4)
        self.forecast_days = st.sidebar.slider("Jours à prédire", 7, 30, 21)

        self.df = load_data(self.ticker, self.years)
        self.pre = GRUPreprocessor()
        self.prophet_model = NeuralProphetPredictor()

        if self.menu == "🏠 Accueil":
            self.show_home()
        elif self.menu == "⚙️ Entraînement + Visualisation":
            self.train_and_visualize()
        elif self.menu == "📡 Prédiction + Visualisation":
            self.predict_and_visualize()

    def show_home(self):
        st.title("📊 Dashboard de prévision des actions AMD")
        st.markdown(
            "Bienvenue dans l'application de prévision des cours de l'action AMD basée sur deux modèles puissants : GRU (RNN) et NeuralProphet."
        )
        st.markdown(
            "Utilisez le menu à gauche pour naviguer entre l'entraînement des modèles et la visualisation des prédictions."
        )
        st.dataframe(self.df.tail(10))

    def train_and_visualize(self):
        st.title("⚙️ Entraînement + Visualisation des modèles")

        x, y, _ = self.pre.transform(self.df)
        x_reshaped = self.pre.reshape_input(x)

        if st.button("Lancer l'entraînement GRU"):
            self.gru_model = GRUPredictor(input_shape=(x_reshaped.shape[1], 1))
            progress_bar = st.progress(
                0, text="Initialisation de l'entraînement du modèle GRU..."
            )
            with st.spinner("Entraînement du modèle GRU en cours..."):
                history = self.gru_model.train(
                    x_reshaped, y, return_history=True, progress_callback=progress_bar
                )
            progress_bar.empty()
            self.gru_model.save("gru_model.h5")

            st.subheader("📉 Évolution de la perte sur les données d'entraînement")
            st.line_chart(history.history["loss"])

            # Afficher les dernières vraies valeurs et les valeurs prédites pour comparaison
            st.subheader("📈 Dernières valeurs d'entraînement vs Prédictions")
            y_pred_scaled = self.gru_model.model.predict(x_reshaped)
            y_pred = self.pre.inverse_transform(y_pred_scaled)
            y_true = self.df["Close"].iloc[-len(y) :].values.reshape(-1, 1)
            test_dates = self.df["Date"].iloc[-len(y) :]
            df_test = pd.DataFrame(
                {
                    "Date": test_dates,
                    "Valeur réelle": y_true.flatten(),
                    "Valeur prédite": y_pred.flatten(),
                }
            )
            st.dataframe(df_test.tail(10))

    def predict_and_visualize(self):
        st.title("📡 Prédiction + Visualisation")

        x, _, _ = self.pre.transform(self.df)
        last_sequence = x[-1]

        if st.button("Lancer la prédiction"):
            self.gru_model = load_gru_model()

            with st.spinner(
                "Chargement du modèle GRU et génération des prédictions..."
            ):
                gru_scaled = self.gru_model.predict_next_days(
                    last_sequence, days=self.forecast_days
                )
                self.gru_preds = self.pre.inverse_transform(gru_scaled)

            with st.spinner("Prévision avec NeuralProphet en cours..."):
                self.prophet_result = self.prophet_model.train_predict(
                    self.df, future_days=self.forecast_days
                )

            self.display_charts()

    def display_charts(self):
        future_dates = pd.date_range(
            start=self.df["Date"].iloc[-1] + pd.Timedelta(days=1),
            periods=self.forecast_days,
            freq="B",
        )

        df_pred = pd.DataFrame(
            {
                "Date": future_dates,
                "GRU": self.gru_preds.flatten(),
                "NeuralProphet": self.prophet_result["yhat1"].values,
            }
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Visualisation",
                "📄 Tableau",
                "🌍 Visualisation générale",
                "📊 Évaluation & Export",
            ]
        )

        with tab1:
            fig = px.line(
                df_pred,
                x="Date",
                y=["GRU", "NeuralProphet"],
                title="Prévisions futures",
            )
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Prix", template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.write("### 📅 Prévisions")
            st.dataframe(df_pred)

        with tab3:
            fig_full = px.line()
            fig_full.add_scatter(
                x=self.df["Date"], y=self.df["Close"], name="Historique", mode="lines"
            )
            fig_full.add_scatter(
                x=df_pred["Date"],
                y=df_pred["GRU"],
                name="GRU Forecast",
                mode="lines+markers",
            )
            fig_full.add_scatter(
                x=df_pred["Date"],
                y=df_pred["NeuralProphet"],
                name="NeuralProphet Forecast",
                mode="lines+markers",
            )
            fig_full.update_layout(
                title="Visualisation complète",
                xaxis_title="Date",
                yaxis_title="Prix",
                template="plotly_white",
            )
            st.plotly_chart(fig_full, use_container_width=True)

        with tab4:
            st.subheader("📏 Évaluation des Prédictions")
            st.write(
                "(Note : Ces évaluations sont faites uniquement sur les prédictions, pas sur un jeu de test réel.)"
            )
            mae = mean_absolute_error(df_pred["GRU"], df_pred["NeuralProphet"])
            rmse = np.sqrt(mean_squared_error(df_pred["GRU"], df_pred["NeuralProphet"]))
            st.metric("MAE entre GRU et NeuralProphet", f"{mae:.2f}")
            st.metric("RMSE entre GRU et NeuralProphet", f"{rmse:.2f}")

            st.subheader("💾 Export des données")
            csv = df_pred.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Télécharger les prédictions (.csv)",
                csv,
                "predictions_amd.csv",
                "text/csv",
            )

        st.markdown(
            """
            ---
            Visualisation des modèles RNN et NeuralProphet sur les actions  
            **Source des données** : Yahoo Finance  
            **Modèles** : GRU (RNN) & NeuralProphet
        """
        )


if __name__ == "__main__":
    StockApp()

    st.sidebar.markdown(
        """
        ---
        **Auteur** : TEUGA KETCHA Ulrich \n
        **Version** : 1.0  \n
        **Contact** : github.com/TeKuV  \n
    """
    )

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import tensorflow as tf
import plotly.graph_objs as go
import time

# Load pre-trained model
model_path = '\trained model(.h5)\cnnmodel.h5'
model = tf.keras.models.load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\OJO ABAYOMI MOSES\Downloads\deeplearning-finalyearproject-2bae82cca4f8.json', scope)
client = gspread.authorize(creds)
spreadsheet_key = '1ppw-iYs-yUzqgHCQ_Vgym7Ivm1j_wU0FFl3lDtlfuOI'
sheet_name = 'sensors_data'

# Class labels with advice
class_labels = {
    0: 'N - Non-ectopic beats (normal beat)',
    1: 'S - Supraventricular ectopic beats',
    2: 'V - Ventricular ectopic beats',
    3: 'F - Fusion Beats',
    4: 'Q - Unknown Beats'
}

advice_labels = {
    0: 'Normal beat. Keep maintaining a healthy lifestyle and regular check-ups.',
    1: 'Supraventricular ectopic beat. It is often benign but consult your doctor if you feel palpitations.',
    2: 'Ventricular ectopic beat. This may require medical attention. Please consult a cardiologist.',
    3: 'Fusion beat. Monitor your symptoms and consult with your healthcare provider.',
    4: 'Unknown beat. It is advisable to get further tests for a precise diagnosis.'
}

# Fetch data from Google Sheets
def fetch_data():
    sheet = client.open_by_key(spreadsheet_key).worksheet(sheet_name)
    data = sheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=range(len(data[0])))
    return df

# Prepare input for model prediction
def prepare_input(input_value):
    sequence_length = 187
    input_data = np.full((1, sequence_length, 1), float(input_value), dtype=np.float32)
    return input_data

# Plot ECG data
def plot_ecg(data):
    fig = go.Figure(data=[go.Scatter(y=data, mode='lines', name='ECG Data')])
    fig.update_layout(title='ECG Data', xaxis_title='Time', yaxis_title='Amplitude')
    st.plotly_chart(fig)

# Streamlit app
def main():
    st.set_page_config(page_title="Health Monitoring App", page_icon="❤️", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predictions", "Visualizations", "Sensor Data"])

    # Fetch data
    data = fetch_data()

    # Sidebar for selecting rows
    st.sidebar.subheader("Select Row for Prediction")
    row_options = list(data.index)
    selected_row = st.sidebar.selectbox("Choose row index:", row_options)

    if page == "Predictions":
        st.title("ECG Predictions")
        st.markdown("### Automated ECG Beat Classification")

        last_row_index = data.index[-1]  # Automatically select the last row index for prediction
        input_value = data.iloc[last_row_index][3]  # Assuming the ECG data is in the 4th column (index 3)
        input_data = prepare_input(input_value)
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]
        advice = advice_labels[predicted_class]

        # Animated loading spinner
        with st.spinner("Predicting..."):
            time.sleep(2)

        st.success(f"Prediction for last row (index {last_row_index}): **{predicted_label}**")
        st.info(f"Advice: {advice}")

        st.markdown("### Predict Other Rows")
        if st.sidebar.button("Predict Selected Row"):
            input_value = data.iloc[selected_row][3]
            input_data = prepare_input(input_value)
            predictions = model.predict(input_data)
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_labels[predicted_class]
            advice = advice_labels[predicted_class]

            st.success(f"Prediction for selected row (index {selected_row}): **{predicted_label}**")
            st.info(f"Advice: {advice}")

        st.markdown("### Manual Prediction")
        manual_input = st.number_input("Enter ECG value:")
        if st.button("Predict Manually"):
            input_data = prepare_input(manual_input)
            predictions = model.predict(input_data)
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_labels[predicted_class]
            advice = advice_labels[predicted_class]

            st.success(f"Manual Prediction: **{predicted_label}**")
            st.info(f"Advice: {advice}")

    elif page == "Visualizations":
        st.title("Visualizations")
        st.markdown("### ECG Data Visualization")
        plot_ecg(data[3].astype(float))  # Plot ECG data

        st.markdown("### Temperature Data for Last Row")
        temp_icon = "🌡️"
        last_row_index = data.index[-1]  # Get the last row index
        last_temp = data.iloc[last_row_index][4]  # Assuming temperature is in the 5th column
        st.write(f"{temp_icon} Temperature: {last_temp} °C")

        st.markdown("### Pulse Data")
        st.line_chart(data[5].astype(float))  # Plot pulse data

    elif page == "Sensor Data":
        st.title("Sensor Data")
        st.markdown("### Raw Sensor Data from Google Sheets")
        st.dataframe(data)

        st.markdown("### Summary Statistics")
        st.write(data.describe())

    # Adding animations and colors for better interaction
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .css-18e3th9 {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        .st-bq {
            color: #ff4b4b;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.balloons()  # Add a balloon animation for fun

if __name__ == "__main__":
    main()

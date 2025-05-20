import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


#
def get_clean_data():
    data = pd.read_csv("data/data.csv")

    data = data.drop(["Unnamed: 32", "id"], axis=1)

    # Encode the diagnosis column with map() function. Assign M to 1 and B to 0.
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


# Define the sidebar
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    # Columns names & labels
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # Create a dictionary input
    input_dict = {}

    # Creating a key/value pair, where key - name of the column, value - value that the user input in the slider
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),  # Change the int to float to fit the rest of the data
            max_value=float(data[key].max()),  # Takes max data from the data
            value=float(data[key].mean()),  # Takes average value from the data
        )
    return input_dict


# Function to scale values
# Takes input_dict with all the values, loops through the data for the max and min, perform some operation that will always be between 0 and 1.
def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(["diagnosis"], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value  # Replace key for scaled value

    return scaled_dict


# Function to show the chart
def get_radar_chart(input_data):

    # All the data treated here between 0 and 1 now
    input_data = get_scaled_values(input_data)

    categories = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[
                # Returns key: value pairs for 'mean'
                input_data["radius_mean"],
                input_data["texture_mean"],
                input_data["perimeter_mean"],
                input_data["area_mean"],
                input_data["smoothness_mean"],
                input_data["compactness_mean"],
                input_data["concavity_mean"],
                input_data["concave points_mean"],
                input_data["symmetry_mean"],
                input_data["fractal_dimension_mean"],
            ],
            theta=categories,
            fill="toself",
            name="Mean Value",  # Name of the trace for mean
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_se"],
                input_data["texture_se"],
                input_data["perimeter_se"],
                input_data["area_se"],
                input_data["smoothness_se"],
                input_data["compactness_se"],
                input_data["concavity_se"],
                input_data["concave points_se"],
                input_data["symmetry_se"],
                input_data["fractal_dimension_se"],
            ],
            theta=categories,
            fill="toself",
            name="Standard Error",  # Name of the trace standard error
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_worst"],
                input_data["texture_worst"],
                input_data["perimeter_worst"],
                input_data["area_worst"],
                input_data["smoothness_worst"],
                input_data["compactness_worst"],
                input_data["concavity_worst"],
                input_data["concave points_worst"],
                input_data["symmetry_worst"],
                input_data["fractal_dimension_worst"],
            ],
            theta=categories,
            fill="toself",
            name="Worst Value",  # Name of the trace for the worst value
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1]  # Range 0 to 1 for the chart values
            )
        ),
        showlegend=True,  # Make this true to show the fields(Mean value, Standard error, Worst Value)
    )
    return fig  # Use streamlit return fig instead to show the chart
    # fig.show() # Renders the plotly chart


def add_predictions(input_data):
    # Import the model from pickle
    model = pickle.load(open("model/model.pkl", "rb"))
    # Import the scaler from pickle
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    # Convert the dict data (key: value, where the key is name of the parameter(e.g. radius mean)) into a single array with the values
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(
        input_array
    )  # Set the default value to the minimum
    prediction = model.predict(input_array_scaled)

    # Add subheader
    st.subheader("Cell cluster prediction")
    st.write("The call cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write(
        "Probability of being bening: ", model.predict_proba(input_array_scaled)[0][0]
    )
    st.write(
        "Probability of being malicious: ",
        model.predict_proba(input_array_scaled)[0][1],
    )

    st.write(
        "This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis."
    )

    # st.write(prediction) # Export the prediction


def main():
    # Set page configuration for the app
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Imports the file into the website with markdown() function
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    # Layout
    # Add the Sidebar that returns the input data in a dictionary
    input_data = add_sidebar()
    # st.write(input_data) # For test purposes - output the input_data with key/value pairs

    # Container for the main part( in order to write inside it - use 'with')
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write(
            "Please connect this app to your cytology lab to help to diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast cancer is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the meassurements by hand using the skuders in the sidebar."
        )

    # First column contains the chart and the second - the prediction box
    # Takes the number of columns and the ratio (first column is going to be 4 times bigger than the 2d)
    col1, col2 = st.columns([4, 1])

    # Write inside of the columns using 'with'
    # Create a radar chart in col1
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        # st.write("Column 1")
    # Write the predictions in col2
    with col2:
        add_predictions(input_data)
        # st.write("Column 2")


if __name__ == "__main__":
    main()

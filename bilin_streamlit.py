import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streng.tools.bilin import Bilin
import seaborn as sns
sns.set_style("whitegrid")


st.write("""
# Bilin App

This app generates bilinear approximations of *force* - *deformation* curves

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# imput_file_delimited = r'P:\tmp\bilin_examples/Example3'
imput_file_delimited = r'examples/Example2'

chart_title = st.sidebar.text_input('chart_title', value='Διγραμμικοποίηση καμπύλης αντίστασης')
xtarget = st.sidebar.number_input('xtarget', value=0.35)


st.write("xtarget = ", xtarget)

bl = Bilin(xtarget=xtarget)

bl.curve_ini.load_delimited(imput_file_delimited, ' ')
# bl.load_space_delimited(r'D:/MyBooks/TEI/RepairsExample/sapfiles/fema/PushoverCurve_modal.txt', ' ')
bl.calc()

f, ax = plt.subplots(figsize=(8, 5))
ax.plot(bl.curve_ini.x, bl.curve_ini.y, label="Initial Curve", lw=2)
ax.plot(bl.bilinear_curve.d_array, bl.bilinear_curve.a_array, label="Bilinear Curve", lw=2)
ax.set_title(chart_title)
ax.set_ylabel('V (kN)')
ax.set_xlabel('δ (m)')
ax.legend()
fig = (f, ax)
# plt.show()
st.write(f)

st.markdown(bl.bilinear_curve.all_quantities)

# # Collects user input features into dataframe
# uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)
# else:
#     def user_input_features():
#         island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
#         sex = st.sidebar.selectbox('Sex',('male','female'))
#         bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
#         bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
#         flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
#         body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
#         data = {'island': island,
#                 'bill_length_mm': bill_length_mm,
#                 'bill_depth_mm': bill_depth_mm,
#                 'flipper_length_mm': flipper_length_mm,
#                 'body_mass_g': body_mass_g,
#                 'sex': sex}
#         features = pd.DataFrame(data, index=[0])
#         return features
#     input_df = user_input_features()

# # Combines user input features with entire penguins dataset
# # This will be useful for the encoding phase
# penguins_raw = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
# penguins = penguins_raw.drop(columns=['species'], axis=1)
# df = pd.concat([input_df,penguins],axis=0)

# # Encoding of ordinal features
# # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sex','island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
# df = df[:1] # Selects only the first row (the user input data)

# # Displays the user input features
# st.subheader('User Input features')

# if uploaded_file is not None:
#     st.write(df)
# else:
#     st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
#     st.write(df)

# # Reads in saved classification model
# load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# # Apply model to make predictions
# prediction = load_clf.predict(df)
# prediction_proba = load_clf.predict_proba(df)


# st.subheader('Prediction')
# penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
# st.write(penguins_species[prediction])

# st.subheader('Prediction Probability')
# st.write(prediction_proba)

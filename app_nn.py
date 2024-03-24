import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPRegressor
import joblib

# # Load the dataset
data = pd.read_csv("Model-2.csv")

# Define categorical columns for one-hot encoding
categorical_cols = ['Kharif', 'Rabi', 'Type']

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='error')
X_encoded = encoder.fit_transform(data[categorical_cols])

# Create DataFrame with encoded features
X = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))

method= data[['Method1', 'Method2', 'Method3']]
mlb = MultiLabelBinarizer()
encoded_m = mlb.fit_transform(method.values.tolist())
m_df = pd. DataFrame(encoded_m, columns= mlb.classes_)
# Concatenate DataFrames along columns axis (axis=1)
merge = pd.concat([X, m_df], axis=1)


# Add numerical columns to X
merge[['Acres', 'Credits']] = data[['Acres', 'Credits']]

merge['credit_ratio']= merge['Credits']/merge['Acres']


Y=merge.iloc[:,-12:]
Y.drop(columns=['Acres','Credits','credit_ratio'], inplace=True)
X=merge.iloc[:,:-12]
# X['Acres'] = merge['Acres']
# X.to_csv('classifier_x.csv', index=False)
# Y.to_csv('classifier_y.csv', index=False)
# X=pd.read_csv('classifier_x.csv')
# Y=pd.read_csv('classifier_y.csv')

# Standardize 'Acres' column in dataframe X
# scaler_X = MinMaxScaler()
# X['Acres'] = scaler_X.fit_transform(X[['Acres']])

# Load pre-trained neural network model
if 'trained_model.joblib' not in st.session_state:
    model = MLPClassifier(hidden_layer_sizes=(30, 20), activation='relu', solver='adam', max_iter=3500)

    # Train the model
    model.fit(X, Y)
    joblib.dump(model, 'trained_model.joblib')
    st.session_state['trained_model.joblib'] = True
    
else:
    model = joblib.load('trained_model.joblib')
    

    
merge['credit_ratio']= merge['Credits']/merge['Acres']
merge.to_csv('one_hot_encoded.csv', index=False)
Yr=merge.iloc[:,-1]
Xr=merge.iloc[:,:-12]
Xr['Acres'] = merge['Acres']
# Standardize 'Acres' column in dataframe X
scaler_Xr = MinMaxScaler()
Xr['Acres'] = scaler_Xr.fit_transform(Xr[['Acres']])

scaler_Yr = MinMaxScaler()

Yr_scaled = scaler_Yr.fit_transform(Yr.values.reshape(-1, 1))
Yr=pd.Series(Yr_scaled.flatten())
# Make predictions on the testing data

if 'trained_reg.joblib' not in st.session_state:
    reg = MLPRegressor(hidden_layer_sizes=(30, 10), activation='relu', solver='adam', max_iter=4500)

    # Train the model
    reg.fit(Xr, Yr)
    joblib.dump(reg, 'trained_reg.joblib')
    st.session_state['trained_reg.joblib'] = True
    
else:
    reg = joblib.load('trained_reg.joblib')

# Load pre-trained neural network model

# Train the model



# Create Streamlit app
st.title('Carbon Credits Prediction')

# Dropdown for choosing Kharif Crop
kharif_crop = st.selectbox('Choose Kharif Crop', data['Kharif'].unique())

# Dropdown for choosing Rabi Crop
rabi_crop = st.selectbox('Choose Rabi Crop', data['Rabi'].unique())

# Dropdown for choosing the states
states = st.selectbox('Choose State', data['Location'].unique())

# Dropdown for choosing the states
# soil = st.selectbox('Choose Soil Type', df['Type'].unique())
soil = st.text_area("Soil Type in This Area", data.loc[data.Location==states]['Type'].iloc[0])

# Text input for entering the area of land in acres
land_area_acres = st.number_input('Enter Area of Land (in Acres)', min_value=0.0, step=1.0)

# Button to trigger prediction
if st.button('Predict'):
    # Convert user input to DataFrame
    user_input = pd.DataFrame({
        'Kharif': [kharif_crop],
        'Rabi': [rabi_crop],
        'Acres': [land_area_acres],
        'Type': [soil]  
    })
    
    # Encode categorical variables
    # for column in ['Location', 'Kharif', 'Rabi', 'Type']:
    #     user_input[column] = label_encoders[column].transform(user_input[column])
    
    # for column in ['Kharif', 'Rabi', 'Type']:
    #     user_input[column] = label_encoders[column].transform(user_input[column])
    # # Standardize features
    # user_input_scaled = scaler_X.transform(user_input)
    # example_input = {
    # 'Kharif': 'Rice',
    # 'Rabi': 'Wheat',
    # 'Acres': 4942100,
    # 'Type': 'Luvisols'
    # }
    
    # user_input = pd.DataFrame([example_input])
    
    example_input_list = encoder.transform(user_input[categorical_cols])
    # Create DataFrame with encoded features
    input_df = pd.DataFrame(example_input_list, columns=encoder.get_feature_names_out(categorical_cols))

    # input_df[['Acres']] = user_input[['Acres']]
    # # Standardize features
    # input_df['Acres'] = scaler_X.transform(input_df[['Acres']])
    predicted_output = model.predict(input_df)
    
    input_df[['Acres']] = user_input[['Acres']]
    # Standardize features
    input_df['Acres'] = scaler_Xr.transform(input_df[['Acres']])
    
    cred = reg.predict(input_df)
    cred=scaler_Yr.inverse_transform(cred.reshape(-1, 1))
    predicted_credits = cred.flatten()*user_input['Acres']
    
    mlb.inverse_transform(predicted_output)
    
    

    op=pd.DataFrame(mlb.inverse_transform(predicted_output))
    op['credits'] = predicted_credits
    
    # Display the predicted output
    st.subheader('Predicted Output:')
    st.write(op)
import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

def set_sidebar():
    st.sidebar.header('Dimentions')
    with open('model/minmax.pkl','rb') as f:
        sidebarcontent=pickle.load(f)
    
    input_dict={}
    
    for label, stat in sidebarcontent.iterrows():
        mi=stat['min']
        ma=stat['max']
        avg=stat['mean']
        input_dict[label]=st.sidebar.slider(
            label=label,
            min_value=0.0,
            max_value=ma,
            value=avg,
            
        )
    return input_dict

def showPredictions(input_data):
    input_arr=np.array(list(input_data.values())).reshape(1,-1)#we are passing list because np.array expects a list, tuple or np array as an argument if its not done the the type will be dict_values
    with open('model/modelv1.pkl','rb') as f:
        model=pickle.load(f)
    with open('model/scalerv1.pkl','rb') as f:
        scaler=pickle.load(f)
    
    st.subheader('Cell Cluster Prediction')
    
    input_arr_scaled=scaler.transform(input_arr)
    prediction=model.predict(input_arr_scaled)

    st.write("The cell cluster is:")

    if prediction[0]==0:
        st.write("Benign")
    else:
        st.write("Malignant")

    st.write("Probabily of being benign: ", round(model.predict_proba(input_arr_scaled)[0][0],4)*100,"%")
    st.write("Probabily of being malignant: ",round(model.predict_proba(input_arr_scaled)[0][1],4)*100,"%")

def getscaledinputdata(input_data):
    with open('model/minmax.pkl', 'rb') as f:
        minmax = pickle.load(f)

    scaled_data = {}

    for key, value in input_data.items():
        min_val = minmax.loc[key, "min"]
        max_val = minmax.loc[key, "max"]

        # simple min-max scaling
        scaled = (value - min_val) / (max_val - min_val)
        scaled_data[key] = scaled

    return scaled_data

def showchart(input_data):
  input_data = getscaledinputdata(input_data)

  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    legend=dict(
        x=0,
        y=0,
        xanchor="left",
        yanchor="top"
    ),
    showlegend=True
)
  
  return fig

def main():
    st.set_page_config(
        page_title='Breast cancer v1',
        page_icon=':female-doctor:',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    input_data=set_sidebar()
    with st.container():
        st.title('Breast Cancer Predictor')
        st.write('Predictions are based on cell cluster propertiest, there dimentions, textures etc, please note that the predictions are being made based on Kaggle dataset, and the model used for predicting the same is Logestic regression model.')

    col1,col2=st.columns([5,2])

    with col1:
        fig=showchart(input_data)
        st.plotly_chart(fig)
    with col2:
        showPredictions(input_data)

if __name__ == '__main__':
    main()
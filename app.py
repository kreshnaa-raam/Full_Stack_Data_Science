import streamlit as st
import pandas as pd
import pickle
import numpy as np

mlmodel = pickle.load(open('model.sav', 'rb'))


def predict(model, input_df):
    predictions_df = model.predict(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))

    st.sidebar.info('Enter Flight Detail Below')
    st.title("Flight Delay Prediction App")

    if add_selectbox == 'Online':
        FlightDate = st.date_input("FlightDate")
        FlightDate = pd.to_datetime(FlightDate,format='%Y/%m/%d')
        day = FlightDate.day
        month = FlightDate.month
        year = FlightDate.year
        DepTime = st.time_input("DepTime")
        #DepTime = pd.to_datetime(DepTime, format='%H:%M').dt.time
        hour = pd.to_datetime(DepTime, format='%H:%M',errors='ignore')
        minutes = pd.to_datetime(DepTime, format='%H:%M',errors='ignore')
        UniqueCarrier = st.text_input("UniqueCarrier")
        Origin = st.text_input("Origin")
        Dest = st.text_input("Dest")
        Distance = st.number_input("Distance")
        Day_of_Week = st.text_input("Day_of_Week")

        output = ""
        input_dict = {'FlightDate': FlightDate,
                      'DepTime': DepTime,
                      'UniqueCarrier': UniqueCarrier,
                      'Origin': Origin,
                      'Dest': Dest,
                      'Distance': Distance,
                      'Day_of_Week': Day_of_Week,
                      'day': day,
                      'month': month,
                      'year': year,
                      'hour': hour,
                      'minutes': minutes
                      }
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            pred = mlmodel.predict(X = input_df)
            if pred[0] == 0:
                output = str("The Flight is not Delayed")
            else:
                output = str("The Flight is Delayed")

        st.success(output)

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            output = mlmodel.predict(X= data)
            st.write(output)


if __name__ == '__main__':
    run()

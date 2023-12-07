import streamlit as st
import requests
import json
import time
from PIL import Image
import os 

st.set_page_config(
    page_title='',
    layout='centered',
    page_icon=None,
    initial_sidebar_state='auto',
)   

IMAGE_PATH = 'images/logo.png'
IMAGE_PATH2 = 'images/car1.jpg'
IMAGE_PATH3 = 'images/car2.jpg'
IMAGE_PATH4 = 'images/car3.jpg'
IMAGE_PATH5 = 'images/car5.jpg'
backend_url = os.getenv('BACKEND_URL')


def send_request(k,v):
    # Replace this URL with the endpoint of your machine learning model
    url = backend_url + "/predict_car_price"
    data = dict(zip(k,v))
    response = requests.post(url, json=data)
    if response.status_code == 200:  # pylint: disable=no-else-return
        return response.json()
    else:
        raise Exception(f"Status: {response.status_code}")

def app():
    
    st.markdown(
        "<h3 style='text-align: center; color: Grey;'>PriceMyRide: Revealing the True Value of Your Used Car</h3>",
        unsafe_allow_html=True,
    )

    col1 ,mid, col2 = st.columns([55,6,35])

    regions = [
    'vermont', 'columbus', 'albany', 'minneapolis / st paul', 'rochester',
    'jacksonville', 'grand rapids', 'philadelphia', 'pittsburgh', 'boston',
    'central NJ', 'madison', 'detroit metro', 'south jersey',
    'raleigh / durham / CH', 'cleveland', 'reno / tahoe', 'charlotte',
    'kansas city, MO', 'orange county', 'omaha / council bluffs', 'louisville',
    'anchorage / mat-su', 'new hampshire', 'indianapolis', 'phoenix',
    'des moines', 'stockton', 'long island', 'rhode island', 'atlanta',
    'dallas / fort worth', 'maine', 'sacramento', 'oklahoma city', 'SF bay area',
    'nashville', 'new york city', 'greenville / upstate', 'san diego',
    'ft myers / SW florida', 'washington, DC', 'ventura county', 'milwaukee',
    'tampa bay area', 'los angeles', 'hartford', 'knoxville',
    'worcester / central MA', 'chicago', 'tulsa', 'st louis, MO', 'north jersey',
    'san antonio', 'albuquerque', 'denver', 'south florida', 'norfolk / hampton roads',
    'colorado springs', 'western massachusetts', 'sarasota-bradenton', 'boise',
    'fresno / madera', 'orlando', 'tucson', 'las vegas', 'cincinnati', 'inland empire',
    'buffalo', 'houston', 'richmond', 'austin', 'modesto', 'el paso', 'baltimore',
    'portland', 'hawaii', "spokane / coeur d'alene", 'redding', 'fayetteville',
    'hudson valley', 'springfield', 'wichita', 'green bay', 'seattle-tacoma',
    'dayton / springfield', 'asheville', 'roanoke', 'eugene', 'medford-ashland',
    'bakersfield', 'new haven', 'akron / canton', 'memphis', 'greensboro',
    'syracuse', 'appleton-oshkosh-FDL', 'birmingham', 'treasure coast', 'daytona beach']

    manufacturers = ['ford', 'gmc', 'chevrolet', 'toyota', 'jeep', 'nissan', 'honda',
       'dodge', 'chrysler', 'ram', 'mercedes-benz', 'infiniti', 'bmw',
       'volkswagen', 'mazda', 'porsche', 'lexus', 'ferrari', 'audi',
       'mitsubishi', 'kia', 'hyundai', 'fiat', 'acura', 'cadillac',
       'rover', 'lincoln', 'jaguar', 'saturn', 'volvo', 'alfa-romeo',
       'buick', 'subaru', 'pontiac', 'mini', 'tesla', 'harley-davidson',
       'mercury', 'datsun', 'land rover', 'aston-martin']
    
    conditions = ['new', 'like new', 'excellent', 'good', 'fair', 'salvage']

    car_cylinders = ['three', 'four', 'five', 'six', 'eight', 'ten', 'tweleve', 'other']

    car_fuels = ['gas', 'diesel', 'other', 'hybrid', 'electric']

    car_drive=["rear-wheel drive (RWD)", "four-wheel drive (4WD)", "front-wheel drive (FWD)"]

    car_type = ['truck', 'pickup', 'other', 'coupe', 'mini-van', 'SUV', 'sedan',
       'offroad', 'convertible', 'hatchback', 'wagon', 'van', 'bus']
    
    colors = ['black', 'silver', 'grey', 'red', 'blue', 'white', 'brown',
       'yellow', 'green', 'purple', 'custom', 'orange']

    image = Image.open(IMAGE_PATH)
    image2 = Image.open(IMAGE_PATH2)
    image3 = Image.open(IMAGE_PATH3)
    image4 = Image.open(IMAGE_PATH4)
    image5 = Image.open(IMAGE_PATH5)

    with col2:
        st.markdown("""***""")
        st.image(image)
        st.image(image2)
        st.image(image3)
        st.image(image4)
        st.image(image5)
        st.markdown("""***""")
    with col1:
        st.markdown("""***""")
        with st.form("predict"):
            region = st.selectbox("Which Region Does Your Car Belong To?",
                                  regions,
                                  placeholder="Select a region",)
            
            #st.write('You selected:', region)

            manufacturer = st.selectbox("What is the Manufacturer of your Car?",
                                  manufacturers,
                                  placeholder="Select a manufacturer",)
            
            #st.write('You selected:', manufacturer)

            condition = st.select_slider("What is the condition of your car?" ,
                                         options=conditions)

            cylinders = st.selectbox("How many cylinders does the engine have?",
                                  car_cylinders,
                                  placeholder="Select the number of cylinders",)
            
            fuel = st.selectbox("What Type of Fuel Does Your Car Use?",
                                  car_fuels,
                                  placeholder="Select the number of cylinders")
            
            odometer= st.number_input("What is Your Car's Odometer Reading?",placeholder="Input the Odometer Reading",step = 1000)

            transmission = st.radio("What Type of Transmission Does Your Car Have?",
                             ["automatic", "manual"])
            
            drive = st.selectbox("What is the Drive Type of Your Car?",
                                  car_drive,
                                  placeholder="Select the Drive Type")
            
            if drive == "rear-wheel drive (RWD)":
                drive = 'rwd'
            elif drive == "four-wheel drive (4WD)":
                drive = '4wd'
            else:
                drive = "fwd"

            typess = st.selectbox("What Type of Vehicle Do You Own?",
                                  car_type,
                                  placeholder="Select the Car type")
            
            paint_color = st.selectbox("What the color of your car?",
                                  colors,
                                  placeholder="Select the Car color")
            
            vehicle_age = st.number_input("How Old is Your Vehicle?",placeholder="Input the Vehicle Age",step = 1)

            keyss = ['region', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'transmission', 'drive', 'type', 'paint_color', 'vehicle_age']
            valuess = [region,manufacturer,condition,cylinders,fuel,odometer,transmission,drive,typess,paint_color,vehicle_age]

            
            if st.form_submit_button('Predict Price'):
                result_json = send_request(keyss,valuess)
                result = result_json['vehicle_cost']
                # result = 1000
                with st.spinner('Wait for it...'):
                     time.sleep(2)
                     st.markdown(f"""
    <h5 style='text-align: center; color: #FAD02E;'>🚗 PriceMyRide Car Valuation 🚗</h5>
    <h6 style='text-align: center; color: #20E0A1;'>Your Car's Estimated Value</h6>
    <h4 style='text-align: center; color: #F65314;'>${result:,}</h4>
    """, unsafe_allow_html=True)

                     st.balloons()                

        st.markdown("""***""")




if __name__ == '__main__':
    app()

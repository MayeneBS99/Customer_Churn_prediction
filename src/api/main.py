from fastapi import FastAPI
from src.api.schemas import Customer_features
from src.api.predictor import Predictor

app = FastAPI()


@app.get('/')
def home_web() :
    """
    API Openning website page
    """
    return {"Message" : "Churn Customer Prediction"}

@app.post('/predict/')
def churn_predict(customer : Customer_features):
    input_data = customer.model_dump()

    predictor_obj = Predictor(input_data)
    prediction = predictor_obj.run()
    return {"The customer is going to leave?" : bool(prediction)}


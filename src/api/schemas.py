from pydantic import BaseModel

class Customer_features(BaseModel):
     CUSTOMER_ID : str
     COLLEGE : bool
     DATA : float
     INCOME : float
     OVERCHARGE : int
     LEFTOVER : int
     HOUSE : float
     LESSTHAN600k : object
     CHILD : int
     JOB_CLASS : int
     REVENUE :float
     HANDSET_PRICE : int
     OVER_15MINS_CALLS_PER_MONTH : int
     TIME_CLIENT : float
     AVERAGE_CALL_DURATION : int
     REPORTED_SATISFACTION : object
     REPORTED_USAGE_LEVEL : object
     CONSIDERING_CHANGE_OF_PLAN : object
     
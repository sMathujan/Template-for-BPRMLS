"""Request/response schemas. Literal types reject bad categories with a clear 422 error."""
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]


class CustomerRecord(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}})

    customerID: Optional[str] = None
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    # Blank for brand-new (tenure 0) customers in the source data
    TotalCharges: Optional[Union[float, str]] = None


class PredictionResponse(BaseModel):
    customerID: Optional[str]
    churn_probability: float
    prediction: int
    status: str
    risk_band: str
    recommended_action: str
    threshold: float
    model_name: str


class BatchRequest(BaseModel):
    records: List[CustomerRecord] = Field(min_length=1, max_length=10_000)


class BatchResponse(BaseModel):
    n_records: int
    n_flagged: int
    predictions: List[PredictionResponse]

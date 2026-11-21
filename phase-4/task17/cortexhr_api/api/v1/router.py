from fastapi import APIRouter

from . import anomaly, attrition, chat, forecast, health, payband, resume

api_v1 = APIRouter(prefix="/v1")
for module in (health, attrition, anomaly, chat, resume, payband, forecast):
    api_v1.include_router(module.router)

# Route aliases matching alternate task specification conventions
api_v1.add_api_route("/attendance/anomaly", anomaly.attendance, methods=["POST"], response_model=anomaly.AnomalyResponse, tags=["Anomaly Detection"], include_in_schema=True)
api_v1.add_api_route("/payroll/anomaly", anomaly.payroll, methods=["POST"], response_model=anomaly.AnomalyResponse, tags=["Anomaly Detection"], include_in_schema=True)
api_v1.add_api_route("/salary/predict", payband.analyze, methods=["POST"], response_model=payband.PayBandResponse, tags=["Pay-Band Analysis"], include_in_schema=True)

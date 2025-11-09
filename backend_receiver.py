# backend_receiver
from fastapi import FastAPI#type:ignore
from pydantic import BaseModel
import uvicorn#type:ignore
import json
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware#type:ignore

# Define the data structure the sensor/mobile will send
class SensorSignal(BaseModel):
    deviceID: str
    problemStatement: str

# Use a temporary file to store the latest signal data
# Streamlit will read this file periodically or when triggered.
DATA_FILE = "signal.json"

app = FastAPI()

origins = [
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/sensor/signal")
async def receive_signal(signal: SensorSignal):
    """
    Receives the HTTP POST signal from the sensor simulator (mobile device).
    """
    try:
        # 1. Save the incoming problem statement to a known location
        data_to_save = {
            "deviceID": signal.deviceID,
            "problemStatement": signal.problemStatement,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        with open(DATA_FILE, 'w') as f:
            json.dump(data_to_save, f)
            
        print(f"Signal received for Device {signal.deviceID}: {signal.problemStatement[:30]}...")
        
        # 2. (Optional: Run the matching logic directly here if you decouple the UI)
        # For now, we'll let Streamlit handle the matching logic.
        
        return {"status": "success", "message": "Signal received and recorded."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Run the server (typically on a port like 8000)
    # You'll need to install: pip install fastapi uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
#HVAC-COND-R4,Critical failure detected: Refrigerant pressure drop and fan motor is vibrating severely. Requires low voltage control diagnostics.
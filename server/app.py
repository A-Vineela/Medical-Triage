from fastapi import FastAPI
from environment import MedicalTriageEnv
from pydantic import BaseModel
import uvicorn

app = FastAPI()

env = None

class StepRequest(BaseModel):
    action_type: str
    payload: dict = {}

@app.post("/reset")
def reset():
    global env
    env = MedicalTriageEnv(task_name="easy_triage")
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(req: StepRequest):
    global env
    action = {
        "action_type": req.action_type,
        "payload": req.payload
    }
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return {"status": "running"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

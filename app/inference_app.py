# major part of code sourced from aws sagemaker example:
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py

import json
import os
import sys
import traceback
import warnings

import pandas as pd
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

import algorithm.utils as utils
from algorithm.model import regressor as model
from algorithm.model_server import ModelServer

prefix = "/opt/ml_vol/"
data_schema_path = os.path.join(prefix, "inputs", "data_config")
model_path = os.path.join(prefix, "model", "artifacts")
failure_path = os.path.join(prefix, "outputs", "errors", "serve_failure")


# get data schema - its needed to set the prediction field name
# and to filter df to only return the id and pred columns
data_schema = utils.get_data_schema(data_schema_path)


# initialize your model here before the app can handle requests
model_server = ModelServer(model_path=model_path, data_schema=data_schema)


# The fastapi app for serving predictions
app = FastAPI()


class NPResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, cls=utils.NpEncoder).encode("utf-8")


@app.get("/ping", tags=["ping", "healthcheck"])
async def ping() -> dict:
    """Determine if the container is working and healthy."""
    response = f"Hello, I am {model.MODEL_NAME} model and I am at you service!"
    return {
        "success": True,
        "message": response,
    }


@app.post("/infer", tags=["inference"])
async def infer(instances: list = Body(embed=True)) -> dict:
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = pd.DataFrame.from_records(instances)
    print(f"Invoked with {data.shape[0]} records")

    # Do the prediction
    try:
        predictions = model_server.predict(data)
        return {
            "predictions": predictions.to_dict(orient="records"),
        }
    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        with open(failure_path, "w") as s:
            s.write("Exception during inference: " + str(err) + "\n" + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during inference: " + str(err) + "\n" + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        return {
            "success": False,
            "message": f"Exception during inference: {str(err)} (check failure file for more details)",
        }


@app.post("/explain", tags=["explain"])
def explain(instances: list = Body(embed=True)) -> dict:
    """Get local explanations on a few samples. In this  server, we take data as JSON, convert
    it to a pandas data frame for internal use and then convert the explanations back to JSON.
    Explanations come back using the ids passed in the input data.
    """
    # Convert from CSV to pandas
    data = pd.DataFrame.from_records(instances)
    print(f"Invoked with {data.shape[0]} records")

    try:
        explanations: dict = model_server.explain_local(data)
        return NPResponse(explanations)
    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        with open(failure_path, "w") as s:
            s.write("Exception during explanation generation: " + str(err) + "\n" + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print(
            "Exception during explanation generation: " + str(err) + "\n" + trc,
            file=sys.stderr,
        )
        # A non-zero exit code causes the training job to be marked as Failed.
        return {
            "success": False,
            "message": f"Exception during explanation: {str(err)} (check failure file for more details)",
        }

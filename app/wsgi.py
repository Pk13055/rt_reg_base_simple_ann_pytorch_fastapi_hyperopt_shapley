import inference_app as myapp

# This is just a simple wrapper for uvicorn to find your app.
# If you want to change the algorithm file, simply change "predictor" above to the
# new file.

app = myapp.app

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="ca7f03ae-3bb4-43fd-8599-d4b1fbafe848",
    resource_group_name="diabetes-rg",
    workspace_name="diabetes-ml-workspace"
)

model = Model(
    path="azureml://jobs/quirky_oil_6rgx9rvsns/outputs/model_output",
    name="diabetes-predictor",
    description="Random Forest model to predict diabetes",
    type="custom_model"
)

registered_model = ml_client.models.create_or_update(model)
print(f"Model registered! ✅")
print(f"Name: {registered_model.name}")
print(f"Version: {registered_model.version}")
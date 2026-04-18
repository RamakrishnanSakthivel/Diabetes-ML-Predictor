from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
    Environment
)
from azure.identity import DefaultAzureCredential
import datetime

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="ca7f03ae-3bb4-43fd-8599-d4b1fbafe848",
    resource_group_name="diabetes-rg",
    workspace_name="diabetes-ml-workspace"
)

# Create a unique endpoint name
endpoint_name = "diabetes-endpoint-" + datetime.datetime.now().strftime("%m%d%H%M")

# Create the endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Diabetes prediction endpoint",
    auth_mode="key"
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"Endpoint created: {endpoint_name} ✅")

try:
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint created: {endpoint_name} ✅")
except Exception as e:
    print(f"Full error: {str(e)}")
    import traceback
    traceback.print_exc()
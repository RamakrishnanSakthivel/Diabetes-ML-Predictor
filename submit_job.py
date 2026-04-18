# submit_job.py
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential

# Connect to your workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="ca7f03ae-3bb4-43fd-8599-d4b1fbafe848",
    resource_group_name="diabetes-rg",
    workspace_name="diabetes-ml-workspace"
)

# Create a compute cluster (free to create, pay only when running)
cpu_cluster = AmlCompute(
    name="diabetes-cluster",
    type="amlcompute",
    size="Standard_DS11_v2",   # Small, cheap VM
    min_instances=0,           # Scales to 0 when idle (saves cost!)
    max_instances=1,
)
ml_client.begin_create_or_update(cpu_cluster).result()
print("Compute cluster ready ✅")

# Define the training job
job = command(
    code=".",                           # Upload current folder
    command="python train.py --model_output ${{outputs.model_output}}",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    compute="diabetes-cluster",
    outputs={"model_output": {}},
    display_name="diabetes-training-run",
    description="Train Random Forest on diabetes dataset"
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted! 🚀")
print(f"View in Studio: {returned_job.studio_url}")
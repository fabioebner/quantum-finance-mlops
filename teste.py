from mlflow.tracking import MlflowClient
import mlflow
import json
from datetime import datetime


mlflow.set_tracking_uri("https://dagshub.com/fabioebner/quantum-finance-mlops.mlflow")

#configuracoes
model_name = "CreditScoreModel"
artifact_relative_path ="models/model.pkl"

client = MlflowClient()

#Buscando as versoes do modelo
version  = client.search_model_versions(f"name='{model_name}'")

ultimo = max(version, key=lambda v: int(v.version))


#Baixando o modelo

dowload_path = client.download_artifacts(
    run_id=ultimo.run_id,
    path=artifact_relative_path,
    dst_path="."
)

model_metadata = {
    "model_name": model_name,
    "version": ultimo.version,
    "run_id": ultimo.run_id,
    "source": ultimo.source,
    "downloaded_at": datetime.now().isoformat()
}

with open("model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=4)
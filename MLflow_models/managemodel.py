from mlflow.tracking import MlflowClient
import mlflow.pyfunc

client = MlflowClient()

# Transitionner la version 1 de "RandomForestModel" à "Staging"
client.transition_model_version_stage(
    name="RandomForestModel",
    version=7,
    stage="Staging"
)

# Charger la version 1 du modèle en stage "Staging"
model = mlflow.pyfunc.load_model(model_uri="models:/RandomForestModel/Staging")

# Utiliser le modèle pour faire des prédictions
from sklearn.datasets import load_iris
iris = load_iris()
X_test = iris.data
predictions = model.predict(X_test)
print(X_test)
print(predictions)

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Entraîner le premier modèle
model_1 = RandomForestClassifier(n_estimators=100)
model_1.fit(X_train, y_train)

# Enregistrer le premier modèle avec MLflow Tracking et Model Registry
with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", 100)
    predictions_1 = model_1.predict(X_test)
    accuracy_1 = accuracy_score(y_test, predictions_1)
    mlflow.log_metric("accuracy", accuracy_1)
    mlflow.sklearn.log_model(model_1, "model")
    result_1 = mlflow.register_model(
        "runs:/"+run.info.run_id+"/model",
        "RandomForestModel"
    )

print("Premier modèle enregistré avec succès.")

# Entraîner le deuxième modèle avec un hyperparamètre différent
model_2 = RandomForestClassifier(n_estimators=200)
model_2.fit(X_train, y_train)

# Enregistrer le deuxième modèle avec MLflow Tracking et Model Registry
with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", 200)
    predictions_2 = model_2.predict(X_test)
    accuracy_2 = accuracy_score(y_test, predictions_2)
    mlflow.log_metric("accuracy", accuracy_2)
    mlflow.sklearn.log_model(model_2, "model")
    result_2 = mlflow.register_model(
        "runs:/"+run.info.run_id+"/model",
        "RandomForestModel"
    )

print("Deuxième modèle enregistré avec succès.")

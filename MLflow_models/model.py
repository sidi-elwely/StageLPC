import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Charger les données
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Entraîner le modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Enregistrer le modèle
mlflow.sklearn.log_model(model, "model")


print("Modèle enregistré avec succès.")

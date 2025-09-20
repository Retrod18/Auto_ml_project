from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

def get_models(task_type: str):
    """
    Returns a dictionary of models appropriate for the given task type.
    This has been expanded to include a wider variety of models.
    """
    if task_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "SVC": SVC(probability=True, random_state=42),
            "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
    elif task_type == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "SVR": SVR(),
            "XGBRegressor": XGBRegressor(random_state=42)
        }
    elif task_type == "clustering":
        models = {
            "KMeans": KMeans(n_clusters=3, random_state=42, n_init=10),
            "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
        }
    elif task_type == "unsupervised": # A more generic unsupervised case
        models = {
            "KMeans": KMeans(n_clusters=3, random_state=42, n_init=10),
            "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
        }
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
    return models


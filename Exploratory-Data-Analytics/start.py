import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def Model(X, Y, title):
    # Develop the Model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3) 
    rg = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs=-1)
    rg.fit(X_train, Y_train)

    print ("\nR2 Score : {:.2f}%".format(r2_score(Y_test, rg.predict(X_test))*100))

    # Showcase the contribution of each feature    
    importances = rg.feature_importances_
    plt.figure(figsize = (10,5))
    values = np.around(importances*100, 1)
    plt.bar(X.columns, values)
    for i in range(X.shape[1]):
        plt.text(i, values[i] + 1, str(values[i]) + '%')
    plt.ylabel(title)
    plt.show()
    return (rg, X_test, Y_test)

def shapAnalysis(model, X, Y):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.initjs()
    shap.summary_plot(shap_values, X)
"""Leakage-safe sklearn model comparison with a final held-out test."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix


def run_comparison(seed=28):
    X,y=make_classification(n_samples=500,n_features=8,n_informative=4,weights=[.7,.3],random_state=seed)
    rng=np.random.default_rng(seed);X[rng.random(X.shape)<.06]=np.nan
    Xdev,Xtest,ydev,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=seed)
    models={'majority':make_pipeline(SimpleImputer(),DummyClassifier(strategy='most_frequent')),
            'logistic':make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=500,random_state=seed)),
            'forest':make_pipeline(SimpleImputer(),RandomForestClassifier(n_estimators=60,max_depth=5,random_state=seed))}
    cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=seed)
    rows=[]
    for name,model in models.items():
        scores=cross_validate(model,Xdev,ydev,cv=cv,scoring=['balanced_accuracy','roc_auc'],return_train_score=False)
        rows.append({'model':name,'cv_balanced_accuracy':float(scores['test_balanced_accuracy'].mean()),'cv_std':float(scores['test_balanced_accuracy'].std()),'cv_roc_auc':float(scores['test_roc_auc'].mean())})
    chosen=max(rows,key=lambda r:r['cv_balanced_accuracy'])['model']
    model=models[chosen].fit(Xdev,ydev);pred=model.predict(Xtest)
    # Test is consulted only after model selection. Repeatedly rerunning to choose seeds is still test leakage.
    return {'comparison':rows,'selected':chosen,'test_n':len(ytest),'test_accuracy':float(accuracy_score(ytest,pred)),
            'test_balanced_accuracy':float(balanced_accuracy_score(ytest,pred)), 'confusion':confusion_matrix(ytest,pred).tolist(),'seed':seed}


if __name__=='__main__':
    import json
    print(json.dumps(run_comparison(),indent=2))

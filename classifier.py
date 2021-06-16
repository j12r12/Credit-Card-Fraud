import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, \
    f1_score, precision_recall_curve, accuracy_score, confusion_matrix, \
    roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as imbpipe
from imblearn.over_sampling import SMOTE

class Classifier:

    def __init__(self, df, desc, clf, params, smote=True, scale=False, scoring="roc_auc"):
        self.df = df
        self.clf = clf
        self.params = params
        self.smote = smote
        self.scale = scale
        self.scoring = scoring
        self.desc = desc

    def create_classifier(self):

        X_cols = self.df.drop("Class", axis=1).columns
        X = self.df.drop("Class", axis=1).values
        y = self.df["Class"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.scale:
            if self.smote:
                pipeline = imbpipe(steps=[["scaler", StandardScaler()],
                                          ["smote", SMOTE(random_state=42)],
                                          ["clf", self.clf]])
                best_clf = GridSearchCV(pipeline, self.params, scoring=self.scoring, cv=3, n_jobs=-1)
                best_clf.fit(X_train, y_train)
                try:
                    feat_imp = best_clf.best_estimator_.steps[2][1].feature_importances_
                    feat_names = X_cols
                except:
                    feat_names = []
                    feat_imp = []
            else:
                pipeline = Pipeline(steps=[("scaler", StandardScaler()),
                                    ("clf", self.clf)])
                best_clf = GridSearchCV(pipeline, self.params, scoring=self.scoring, cv=3, n_jobs=-1)
                best_clf.fit(X_train, y_train)
                try:
                    feat_imp = best_clf.best_estimator_.steps[1][1].feature_importances_
                    feat_names = X_cols
                except:
                    feat_names = []
                    feat_imp = []
        else:
            if self.smote:
                pipeline = imbpipe(steps=[["smote", SMOTE(random_state=42)],
                                          ["clf", self.clf]])
                best_clf = GridSearchCV(pipeline, self.params, scoring=self.scoring, cv=3, n_jobs=-1)
                best_clf.fit(X_train, y_train)
                try:
                    feat_imp = best_clf.best_estimator_.steps[1][1].feature_importances_
                    feat_names = X_cols
                except:
                    feat_names = []
                    feat_imp = []
            else:
                pipeline = Pipeline(steps=[("clf", self.clf)])
                best_clf = GridSearchCV(pipeline, self.params, scoring=self.scoring, cv=3, n_jobs=-1)

                best_clf.fit(X_train, y_train)

                feat_imp = best_clf.best_estimator_.steps[0][1].feature_importances_
                feat_names = X_cols

        return best_clf, X_test, y_test, feat_names, feat_imp

    def output(self):

        best_clf, X_test, y_test, feat_names, feat_imp = self.create_classifier()

        y_pred = best_clf.predict(X_test)
        
        with open(f"{self.clf.__class__.__name__} - {self.desc}.txt", "w") as f:
            f.write(f"*** {self.clf.__class__.__name__}Classifier Scores ***")
            f.write("\n")
            f.write(f"SMOTE:{self.smote}")
            f.write("\n")
            f.write(f"Scaling:{self.scale}")
            f.write("\n")
            f.write(f"Scoring:{self.scoring}")
            f.write("\n")
            f.write(f"Parameters:{best_clf.best_params_}")
            f.write("\n")
            f.write(f"Accuracy:{accuracy_score(y_test, y_pred)}")
            f.write("\n")
            f.write(f"Precision:{precision_score(y_test, y_pred)}")
            f.write("\n")
            f.write(f"Recall:{recall_score(y_test, y_pred)}")
            f.write("\n")
            f.write(f"F1 Score:{f1_score(y_test, y_pred)}")
            f.write("\n")
            f.write(f"ROC_AUC Score:{roc_auc_score(y_test, y_pred)}")
            f.write("\n")
            f.write(f"Confusion Matrix:{confusion_matrix(y_test, y_pred)}")

        feats = pd.DataFrame({"Feature Names":feat_names, "Feature Importances":feat_imp}).sort_values(by="Feature Importances",
                                                                                                       ascending=False)
        fig1, ax1 = plt.subplots()
        ax1.bar(feats["Feature Names"], feats["Feature Importances"])
        ax1.set_title(f"{self.clf.__class__.__name__} - Feature Importances")
        # ax1.set_xticks(rotation=45)
        plt.savefig(f"Features - {self.clf.__class__.__name__} - {self.desc}.png")

        prec, recall, thresh = precision_recall_curve(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        fig, ax = plt.subplots(2,1)
        ax[0].plot(prec, recall)
        ax[0].set_title("Precision/Recall Curve")
        ax[1].plot(fpr, tpr)
        ax[1].set_title("ROC Curve")
        plt.savefig(f"P-R and ROC Curves - {self.clf.__class__.__name__} - {self.desc}.png")

        print("Iteration Complete...")

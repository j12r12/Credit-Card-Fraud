from pre-processing import preprocessing
from classifier import Classifier

if __name__ == "__main__":
  
  df = pd.read_csv("creditcard.csv")
  
  vs = ["V3", "V4", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "V18",
      "Amount", "Class"]

  ns_nl = preprocessing(df, vs, stationary=False, log=False)
  ns_l = preprocessing(df, vs, stationary=False, log=True)
  s_nl = preprocessing(df, vs, stationary=True, log=False)
  s_l = preprocessing(df, vs, stationary=True, log=True)
  
  rf1_params = {"clf__n_estimators":[100], "clf__max_depth":[100]}
  rf1 = Classifier(ns_l, "ns_l", RandomForestClassifier(n_jobs=3, verbose=3), rf1_params, scoring="recall")
  rf1.output()

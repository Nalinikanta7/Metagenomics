import pickle

with open('Model_RF.pickle', 'rb') as f:
    rf = pickle.load(f)

with open('Model_RF.pickle', 'rb') as f:
    clf = pickle.load(f)

x = input("Please enter your phrase: ")
y = clf.predict_proba(rf.transform([x]))
print(y[0])

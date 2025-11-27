import joblib

model = joblib.load('ml_models/toy_models/LRmodel.pkl')

prediction = model.predict(
    ['I am stressed af, and I want to kill everything', 'Dragazo the conqueror']
)
print(prediction)

import pandas as pd
import joblib
import os
import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model


XOR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,0],
}

df = pd.DataFrame(XOR)

print(df)

X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()


save_model(model, filename = "and.model")
save_plot(df, filename = "and.png", model)



import pandas as pd
import joblib
import os
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model

def main(data, eta, epochs, filename, plotFileName):



    df = pd.DataFrame(data)

    print(df)

    X,y = prepare_data(df)



    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()


    save_model(model, filename = "and.model")

    save_plot(df, model ,file_name= "and.png" )

if __name__ == "__main__":

    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    main(data= AND, eta= ETA, epochs= EPOCHS, filename= "and.model", plotFileName= "and.png")

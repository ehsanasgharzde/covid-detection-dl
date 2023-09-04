import tensorflow
import keras
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from typing import Any
import pandas
import numpy


GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
END = "\033[0m"


class ANN:
    def __init__(self) -> None:
        """Initializes the ANN Class.

        Results
        ----------
        model: object
        pre define a variable to store and work with Model.

        stopper: object
        pre define a variable to store and work with Early Stopper.

        checker: object
        pre define a variable to store and work with Model Checkpoints.
        ----------
        
        Parameters:
        ----------
        None, takes no argument.
        ----------

        Raises
        ----------
        None, raises no errors.

        """
        self.model = None
        self.stopper = None
        self.checker = None
        tensorflow.get_logger().setLevel('ERROR')


    def create(self, inputLayers: int = 11, hiddenLayers: int = 250, outputLayers: int = 1, activator: Any = "sigmoid") -> None:
        """Creates a model and sets the input, hidden, and output layers with
        the given activation function.
        
        Results
        ----------
        model: object
        variable to store and work with Model.
        ----------
        
        Parameters:
        ----------
        inputLayers: int -> default: 11, Required
        it is an integer that specifies the number of input layers,
        'how many inputs does a model take?'.

        hiddenLayer: int -> default: 250, Required
        it is an integer that specifies the number of hidden layers,
        'how many hidden layers does a model have?'.

        outputLayer: int -> default: 1, Required
        it is an integer that specifies the number of output layers,
        'how many outputs does a model return?'.

        activator: Any -> default: 'sigmoid', Required
        it is an string, function, or object that specifies the the activation function
        for each and evey node in model's neural networks.
        ----------

        Raises
        ----------
        None, raises no errors.

        """
        self.model = keras.Sequential()

        self.model.add(Dense(hiddenLayers, activation=activator, input_shape=(inputLayers, )))
        for layer in range(hiddenLayers):
            self.model.add(Dense(hiddenLayers, activation=activator))
        
        self.model.add(Dense(outputLayers))


    def compile(self, optimizer: Any = "adamax", loss: Any = "mean_squared_error", metrics: list = ["accuracy", ]) -> None:
        """Sets configurations for the model.

        Results
        ----------
        model: object
        updated model with configuration.
        ----------
        
        Parameters:
        ----------
        optimizer: Any -> default: 'adamax', Required
        optimizer estimator.

        loss: Any -> default: 'mean_squared_error'
        loss function.

        metrics: list -> default: ["accuracy", ], Optional
        list of metrics to be evaluated by the model during training and testing.
        ----------

        Raises
        ----------
        None, raises no errors.

        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def callback(self, limit: int = 250, path: str = "checkpoints/best-epoch-checkpoint", optimizer: str = "adamax", monitor: str = "loss", verbose: int = 1, bestOnly: bool = True, mode: str = "min") -> object:
        """Creates callback object to be used in training processes.

        Results
        ----------
        stopper: object
        sets stopper variable with Early Stopper.

        checker: object
        sets checker variable with Model Checkpoints.
        ----------
        
        Parameters:
        ----------
        limit: int -> default: 3, Required
        number of epochs with no improvement after which training will be stopped.

        path: str -> default: 'checkpoints/best-epoch-checkpoint'
        saves each improvement in path directory.

        optimizer: str -> default: 'adamax'
        for saving checkpoints for specific optimizer.

        monitor: str -> default: 'loss', Required
        prints model epochs loss.

        verbose: int -> default: 1, Optional
        verbosity mode, 0 or 1. Mode 0 is silent, 
        and mode 1 displays messages when the callback takes an action.

        bestOnly: bool -> default: True, Optional
        it only saves when the model is considered the "best" and the latest best model 
        according to the quantity monitored will not be overwritten.

        mode: str -> default: 'min', Optional
        {'auto', 'min', 'max'}
        the decision to overwrite the current save file is made 
        based on either the maximization or the minimization of the monitored quantity.

        'loss' -> 'min', 
        'acc' -> 'max'
        ----------

        Raises
        ----------
        None, raises no errors.

        """        
        path = f"{path}-{optimizer}"

        self.stopper = EarlyStopping(patience=limit, monitor=monitor)
        self.checker = ModelCheckpoint(path, monitor=monitor, verbose=verbose, save_best_only=bestOnly, mode=mode)


    def train(self, trainX: numpy.ndarray, trainY: numpy.ndarray, batch: int = 10, validation: int = None, epochs: int = 250) -> None:
        """trains the model with given dataset.

        Results
        ----------
        model: object
        updated with epochs and has training now.
        ----------
        
        Parameters:
        ----------
        trainX: numpy.ndarray -> default: ..., Required
        inputs as X for model to create a equation with.

        trainY: numpy.ndarray -> default: ..., Required
        outputs as Y for model to update a equation with.

        validiation: int -> default: None, Optional
        used for tuning hyper-parameters.

        epochs: int -> default: 250, Required
        specifies how many times a model trains on a dataset.
        ----------

        Raises
        ----------
        None, raises no errors.

        """
        trainX = numpy.asarray(trainX).astype(numpy.float32)
        trainY = numpy.asarray(trainY).astype(numpy.float32)


        print(f"\n{BOLD}{YELLOW}Model Learning Approach {1}{END}")
        self.model.fit(trainX, trainY, batch_size=batch, validation_split=validation, epochs=epochs, callbacks=[self.stopper, self.checker])
        print(f"{BOLD}{GREEN}[Approach {1}] Completed Successfully.{END}")

    
    def inputs(self) -> pandas.DataFrame:
        """Takes real world test case and processes it.

        Results
        ----------
        testX: pandas.DataFrame
        takes needed values and turns them into a pandas.DataFrame
        so model could work on it.
        ----------
        
        Parameters:
        ----------
        None, takes no argument.

        Raises
        ----------
        None, raises no errors.

        """
        Gender = input("Gender: ").lower()
        testX = pandas.DataFrame({
            "HB": [int(input("HB: "))],
            "HCT": [int(input("HCT: "))],
            "WBC": [int(input("WBC: "))],
            "PLT": [int(input("PLT: "))],
            "CPK": [int(input("CPK: "))],
            "CRP": [float("{:.1f}".format(float(input("CRP: "))))],
            "ESR": [int(input("ESR: "))],
            "LDH": [int(input("LDH: "))],
            "Feitine": [int(input("Feitine: "))],
            "D'dimer": [int(input("D'dimer: "))],
            "Interleukine": [int(input("Interleukine: "))],
        })

        return Gender, testX


    def prediction(self, testX: numpy.ndarray) -> float:
        """
        Results
        ----------
        testY: 2 decimal-point rounded float
        result test case that model predicts.
        ----------
        
        Parameters:
        ----------
        testX: numpy.ndarray -> default: ..., Required
        new test case that model must work on.
        ----------

        Raises
        ----------
        None, raises no errors.

        """
        testY = self.model.predict(testX)
        testY = round(float(testY[0][0]), 2)

        return testY

    def save(self, gender: str, testX: numpy.ndarray, testY: numpy.ndarray, path: str = "files/covid-testcase-predicted.csv") -> None:
        """Saves newly tested data into train dataset.

        Results
        ----------
        test: pandas.DataFrame
        merged Gender, testX, and testY together in order of axis=1 (columns).

        predicted: pandas.DataFrame
        contaiting predicted values.

        csv-file: csv
        saved predicted data into path.
        ----------
        
        Parameters:
        ----------
        path: str -> default: 'files/covid-testcase-predicted.csv', Required
        directory that results will be saved in.

        testX: numpy.ndarray -> default: ..., Required
        new test case that model works on. 

        testY: pandas.DataFrame -> default: ..., Required
        result test case that model predicts.
        ----------

        Raises
        ----------
        None, raises no errors.

        """
        Gender = pandas.DataFrame({"Gender": [gender]})
        testX = pandas.DataFrame({
            "HB": [testX[0][0]],
            "HCT": [testX[0][1]],
            "WBC": [testX[0][2]],
            "PLT": [testX[0][3]],
            "CPK": [testX[0][4]],
            "CRP": [testX[0][5]],
            "ESR": [testX[0][6]],
            "LDH": [testX[0][7]],
            "Feitine": [testX[0][8]],
            "D'dimer": [testX[0][9]],
            "Interleukine": [testX[0][10]]
        })

        test = pandas.concat([Gender, testX, testY], axis=1, ignore_index=True)

        try:
            predicted = pandas.read_csv(path, ignore_index=True)
            predicted = pandas.concat([predicted, test], axis=0, ignore_index=True)
        except:
            predicted = test

        predicted.to_csv(path, index=False)


    def load(self, path: str = "checkpoints/best-epoch-checkpoint"):
        """Load previous trained weights for model to use.

        Results:
        ----------
        load weights.

        Parameters:
        ----------
        None, takes no argument.


        Raises
        ----------
        None, raises no errors.
        
        """
        path += "/variables/variables"
        self.model.load_weights(path)

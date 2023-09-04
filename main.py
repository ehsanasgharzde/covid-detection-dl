import pandas
import modeling
import algorithms
import sklearn.preprocessing


LOSS = algorithms.Loss()
OPTIMIZER = algorithms.Optimizer()
METRICS = algorithms.Metrics()

GREEN = "\033[0;32m"
PURPLE = "\033[0;35m"
BOLD = "\033[1m"
END = "\033[0m"

trainXscaler = sklearn.preprocessing.MinMaxScaler()
trainYscaler = sklearn.preprocessing.MinMaxScaler()
sequence = modeling.ANN()

print("\n" * 1000)

train = pandas.read_csv("files/covid-testcase.csv")
try:
    train.drop(["Unnamed: 0"], axis=1)
except:
    pass


trainX = pandas.DataFrame(train, columns=["HB", "HCT", "WBC", "PLT", "CPK", "CRP", "ESR", "LDH", "Feitine", "D'dimer", "Interleukine"])
trainY = pandas.DataFrame(train, columns=["Covid Probability"])

# Preprocessing: MinMaxScaler Normalization Algorithm
trainXscaler.fit(trainX)
trainX = trainXscaler.transform(trainX)

trainYscaler.fit(trainY)
trainY = trainYscaler.transform(trainY)

print(f"{GREEN}[Preprocessing] Completed{END}", "->", f"{PURPLE}Dataset preprocessed successfully.{END}")


if __name__ == '__main__':
    # Step 1: Creating a Artificial Neural Network containing [<input-layer> <hidden-layer> <output-layer>]
    sequence.create()
    print(f"{GREEN}[Creating] Completed{END}", "->", f"{PURPLE}Model created successfully.{END}")

    # Step 2: Set Optimizer, Loss Function, and metrics configuration for the model
    sequence.compile()
    print(f"{GREEN}[Compiling] Completed{END}", "->", f"{PURPLE}Model compiled successfully.{END}")

    # Step 4: Set callback methods for the model
    sequence.callback()
    print(f"{GREEN}[Setting] Completed{END}", "->", f"{PURPLE}Model callbacks set successfully.{END}")

    # Step 4: Training the model with train dataset (If loading required use [Ctrl + /] on line 56, 57, 60, and 61)
    # sequence.train(trainX=trainX, trainY=trainY)
    # print(f"{GREEN}[Training] Completed{END}", "->", f"{PURPLE}Model trained successfully{END}")

    # Step 4: Loading checkpoints to be used for the model (If training required use [Ctrl + /] on line 56, 57, 60, and 61)
    sequence.load(path="checkpoints/best-epoch-checkpoint-adamax")
    print(f"{GREEN}[Loading] Completed{END}", "->", f"{PURPLE}Checkpoints loaded successfully.{END}")

    choice = "yes"
    while choice in ["y", "yes"]:
        # Step 5: Getting test case from user
        Gender, testX = sequence.inputs()

        testX = trainXscaler.transform(testX)
        print(f"{GREEN}[Processing] Completed{END}", "->", f"{PURPLE}Inputs processed successfully.{END}")

        # Step 6: Use model to predict the results fir testX
        testY = sequence.prediction(testX=testX)

        print(f"{GREEN}[Predicting] Completed{END}", "->", f"{PURPLE}Prediction made successfully.{END}")

        # Step 7: Processing and printing the results
        print(f"Covid Probability: {testY}")
        testY = pandas.DataFrame({"Covid Probability": [testY]})
        print(f"{GREEN}[Framing] Completed{END}", "->", f"{PURPLE}Test case dataframe created successfully.{END}")

        # Step 8: Save the test case into covid-testcase.csv (need debug)
        sequence.save(gender=Gender, testX=testX, testY=testY)
        print(f"{GREEN}[Saving] Completed{END}", "->", f"{PURPLE}Test case saved successfully.{END}")

        # Step 9: Ask for program to continue
        choice = input(f"{GREEN}Do you want to test again? [y/n] {END}").lower()
        

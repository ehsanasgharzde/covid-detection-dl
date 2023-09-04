import keras.losses as KL
import keras.optimizers as KO
import keras.metrics as KM
import pandas


class Loss:
    """Loss Function Algorithms

    Parameters
    ----------
        1. Probabilistic losses:
            1. BinaryCrossentropy: Computes the cross-entropy loss between true labels and predicted labels.
            2. CategoricalCrossentropy: Computes the crossentropy loss between the labels and predictions.
            3. SparseCategoricalCrossentropy: Computes the crossentropy loss between the labels and predictions.
            4. Poisson: Computes the Poisson loss.
            5. KLDivergence: Computes Kullback-Leibler divergence loss.
            ----------


        2. Regression losses:
            1. MeanSquaredError: Computes the mean of squares of errors between labels and predictions.
            2. MeanAbsoluteError: Computes the mean of absolute difference between labels and predictions.
            3. MeanAbsolutePercentageError: Computes the mean absolute percentage error
            4. MeanSquaredLogarithmicError: Computes the mean squared logarithmic error
            5. CosineSimilarity: Computes the cosine similarity between labels and predictions.
            6. Huber: Computes the Huber loss
            7. LogCosh: Computes the logarithm of the hyperbolic cosine of the prediction error.
            ----------
    """

    Probabilistic = pandas.DataFrame({
        "BinaryCrossentropy": [KL.BinaryCrossentropy(from_logits=True)],
        "CategoricalCrossentropy": [KL.CategoricalCrossentropy(from_logits=True)],
        "SparseCategoricalCrossentropy": [KL.SparseCategoricalCrossentropy(from_logits=True)],
        "Poisson": [KL.Poisson(reduction="auto", name="poisson")],
        "KLDivergence": [KL.KLDivergence()],
    })

    Regression = pandas.DataFrame({
        "MeanSquaredError": [KL.MeanSquaredError(reduction='sum_over_batch_size')],
        "MeanAbsoluteError": [KL.MeanAbsoluteError()],
        "MeanAbsolutePercentageError": [KL.MeanAbsolutePercentageError()],
        "MeanSquaredLogarithmicError": [KL.MeanSquaredLogarithmicError()],
        "CosineSimilarity": [KL.CosineSimilarity(axis=1)],
        "Huber": [KL.Huber()],
        "LogCosh": [KL.LogCosh()],
    })


class Optimizer:
    """Optimize Algorithms

    Parameters
    ----------
        1. SGD: Keras Stochastic Gradient Descent Optimizer.
        2. RMSprop: Root mean Square propagation.
        3. Adagrad: Keras uses specific parameters in the learning rates.
        4. Adadelta: Used in scenarios involving adaptive learning rates concerning the gradient descent value.
        5. Adam: Adaptive Moment estimation.
        6. Adamax: Adaption of the Adam optimizer hence the name Adam max.
        7. Nadam: Nesterov and adam optimizer.
        8. Ftrl: Follow The Regularized Leader.
    """

    Estimation = pandas.DataFrame({
        "SGD": [KO.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)],
        "RMSprop": [KO.RMSprop(learning_rate=0.001, rho=0.9)],
        "Adagrad": [KO.Adagrad(learning_rate=0.01)],
        "Adadelta": [KO.Adadelta(learning_rate=1.0, rho=0.95)],
        "Adam": [KO.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)],
        "Adamax": [KO.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)],
        "Nadam": [KO.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)],
        "Ftrl": [KO.Ftrl(learning_rate=0.001, beta=0.0)],
    })


class Metrics:
    """Metric Algorithms

    Parameters
    ----------
        1. Accuracy metrics:
            1. Accuracy: Calculates how often predictions equal labels.
            2. BinaryAccuracy: Calculates how often predictions match binary labels.
            3. CategoricalAccuracy: Calculates how often predictions match one-hot labels.
            4. SparseCategoricalAccuracy: Calculates how often predictions match integer labels.
            5. TopKCategoricalAccuracy: Computes how often targets are in the top K predictions.
            6. SparseTopKCategoricalAccuracy: Computes how often integer targets are in the top K predictions.
       ----------


        2. Probabilistic losses:
            1. BinaryCrossentropy: Computes the crossentropy metric between the labels and predictions.
            2. CategoricalCrossentropy: Computes the crossentropy metric between the labels and predictions.
            3. SparseCategoricalCrossentropy: Computes the crossentropy metric between the labels and predictions.
            4. KLDivergence: Computes Kullback-Leibler divergence metric
            5. Poisson: Computes the Poisson metric
            ----------


        3. Regression losses:
            1. MeanSquaredError: Computes the mean squared error
            2. RootMeanSquaredError: Computes root mean squared error metric
            3. MeanAbsoluteError: Computes the mean absolute error between the labels and predictions.
            4. MeanAbsolutePercentageError: Computes the mean absolute percentage error
            5. MeanSquaredLogarithmicError: Computes the mean squared logarithmic error
            6. CosineSimilarity: Computes the cosine similarity between the labels and predictions.
            7. LogCoshError: Computes the logarithm of the hyperbolic cosine of the prediction error.
            ----------
    """

    Accuracy = pandas.DataFrame({
        "Accuracy": [KM.Accuracy()],
        "BinaryAccuracy": [KM.BinaryAccuracy()],
        "CategoricalAccuracy": [KM.CategoricalAccuracy()],
        "SparseCategoricalAccuracy": [KM.SparseCategoricalAccuracy()],
        "TopKCategoricalAccuracy": [KM.TopKCategoricalAccuracy(k=1)],
        "SparseTopKCategoricalAccuracy": [KM.SparseTopKCategoricalAccuracy(k=1)],
    })

    Probabilistic = pandas.DataFrame({
        "BinaryCrossentropy": [KM.BinaryCrossentropy()],
        "CategoricalCrossentropy": [KM.CategoricalCrossentropy()],
        "SparseCategoricalCrossentropy": [KM.SparseCategoricalCrossentropy()],
        "KLDivergence": [KM.KLDivergence()],
        "Poisson": [KM.Poisson()],
    })

    Regression = pandas.DataFrame({
        "MeanSquaredError": [KM.MeanSquaredError()],
        "RootMeanSquaredError": [KM.RootMeanSquaredError()],
        "MeanAbsoluteError": [KM.MeanAbsoluteError()],
        "MeanAbsolutePercentageErro": [KM.MeanAbsolutePercentageError()],
        "MeanSquaredLogarithmicError": [KM.MeanSquaredLogarithmicError()],
        "CosineSimilarity": [KM.CosineSimilarity(axis=1)],
        "LogCoshError": [KM.LogCoshError()],
    })

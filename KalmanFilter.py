import numpy as np
from numpy.linalg import inv


positionObserved = np.array([4000, 4260, 4550, 4860, 5110])
velocityObserved = np.array([280, 282, 285, 286, 290])

a = np.array([2])
x_init = positionObserved[0]
v_init = velocityObserved[0]
t = 1

stateMatrix = np.array([[1, t], [0, 1]])
controlMatrix = np.array([[1 / 2 * (t ** 2)], [t]])

# Process errors from the process covaraince matrix
x_cvError = 20
v_cvError = 5

# Observation errors
x_obError = 25
v_obError = 6


def GeneratePredictedState(i):
    stateVector = np.array([[positionObserved[i]], [velocityObserved[i]]])
    vec1 = np.matmul(stateMatrix, stateVector)
    vec2 = a * controlMatrix
    newState = np.add(vec1, vec2)
    return newState


predicted_state = GeneratePredictedState(0)


def GenerateProcessCovariance(positionError, velocityError):
    return np.array([[positionError ** 2, 0], [0, velocityError ** 2]])

initProcessCovMatrix = GenerateProcessCovariance(x_cvError, v_cvError)


def GeneratePredictedProcessCovariance():
    x = np.matmul(stateMatrix, initProcessCovMatrix)
    y = np.matmul(x, np.transpose(stateMatrix))
    return np.diag(np.diag(y)) #Approximating non-diagonal terms to 0

processCovMatrix = GeneratePredictedProcessCovariance()

H = np.identity(2)
def KalmanGain(positionObvError, velObvError):
    obvError = np.array([[positionObvError**2, 0], [0, velObvError**2]])
    numerator = np.matmul(processCovMatrix, np.transpose(H))
    denominator = np.matmul(np.transpose(H), processCovMatrix)
    denominator = np.matmul(denominator, np.transpose(H))
    denominator = np.add(denominator, obvError)
    return np.matmul(numerator, inv(denominator))

GainMatrix = KalmanGain(x_obError, v_obError)

nextObservation = np.array([[positionObserved[1]], [velocityObserved[1]]])

def calculateCurrentState():
    currentState = nextObservation - predicted_state  # Note that since H is the identity matrix, I have simplified the calculation
    currentState = np.matmul(GainMatrix, currentState)
    currentState = predicted_state + currentState
    return currentState

current_state = calculateCurrentState()

def updateCovMatrix():
    update = np.identity(2) - GainMatrix #Similar simplification of H
    update = np.matmul(update, processCovMatrix)
    return update

updatedMatrix = updateCovMatrix()

# Notes

- Check other hyperparameter tuning methods beyond GridSearch

svm-2: "During the following experiments ... one specific parameter is varied while all the others are fixed on a certain value."

## Parameter breakdown

### MLP

Required non default:
- activation='relu' -> 'tanh'
- random\_state=None -> 0
- shuffle=True -> False

-------------------------------

Relevant:
- hidden\_layer\_sizes=(100,) - hidden nodes -> {3-7}, {10},{4-20} -> [1-20]
- solver='adam'
- alpha=0.0001
- tol=0.0001

Other:
- verbose=False
- max\_iter=200 -> 5000

-------------------------------

Optimizer specific (irrelevant):
- learning\_rate='constant'
- learning\_rate\_init=0.001
- batch\_size='auto'
- power\_t=0.5
- max\_fun=15000
- momentum=0.9
- nesterovs\_momentum=True
- early\_stopping=False
- validation\_fraction=0.1
- beta\_1=0.9
- beta\_2=0.999
- epsilon=1e-08
- n\_iter\_no\_change=10
- warm\_start=False


### SVR

-------------------------------

Relevant:
- kernel='rbf'
- tol=0.001
- C=1.0
- epsilon=0.1

Optimizer specific (relevant):
- degree=3
- gamma='scale'
- coef0=0.0

Other:
- verbose=False

-------------------------------

Irrelevant:
- shrinking=True
- cache\_size=200
- max\_iter=-1

### RF

Required non default:
- random\_state=None -> 0

-------------------------------

Relevant:
- n\_estimators=100
- max\_depth=None
- max\_features=1.0

Other:
- oob\_score=False -> mse
- n\_jobs=None
- verbose=False

-------------------------------

Irrelevant:
- criterion='squared\_error'
- bootstrap=True
- warm\_start=False
- ccp\_alpha=0.0
- max\_leaf\_nodes=None
- min\_impurity\_decrease=0.0
- min\_weight\_fraction\_leaf=0.0
- max\_samples=None
- min\_samples\_split=2
- min\_samples\_leaf=1


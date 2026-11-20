##traning and evaluvating using Linear_regression
data ={
    'study_hours':[
        2,3,4,5,6,7,8,9,2,4,5,6,7,3,9,4,6,7,8,8
    ],
    'attendance':[
        60,65,70,75,80,85,90,95,62,72,76,82,86,92,68,96,74,84,88,94
    ],
    'assignment_score':[
        50,55,60,65,70,75,80,90,52,62,67,72,78,85,58,92,64,74,79,88
        ],
    'exam_marks':[
        45,50,55,62,70,78,85,92,46,57,64,72,80,89,53,95,60,75,83,91
    ]
}

df = pd.DataFrame(data)
print(df)

print("first five records:")
print(df.head())

print("\nStatistics of the dataset:")
print(df.describe())

print("\nMissing values :")
print(df.isnull().sum())

df['study_attendance']=(df['study_hours']*df['attendance']/100)
print("data after feature engineering:")
print(df.head())

x = df[[
    'assignment_score',
    'attendance',
    'study_attendance'
]]
y = df['exam_marks']

print("input features:")
print(x.head())

print("\ntarget variable:")
print(y.head())
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.40, random_state=42
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.50, random_state=42
)

print("training records:", len(x_train))
print("validation records:", len(x_val))
print("testing records:", len(x_test))

model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

model.fit(x_train, y_train)
print("Model trained successfully!")

val_predicition = model.predict(x_val)
print("actual validation marks:")
print(list(y_val))

print("\npredicated validation marks:")
print(np.round(val_predicition,2))

val_r2 = r2_score(y_val, val_predicition)
print("\nvalidation R2 score:", round(val_r2,3))

test_prediction = model.predict(x_test)
print("\nactual test marks:")
print(list(y_test))

print("\npredicated test marks:")
print(np.round(test_prediction,2))
#calculate performance metrics
mae = mean_absolute_error(y_test, test_prediction)
mse = mean_squared_error(y_test, test_prediction)

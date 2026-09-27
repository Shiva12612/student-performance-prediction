import pandas as pd
import joblib


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

regression_model = joblib.load(
    "models/regression_model.pkl"
)

classification_model = joblib.load(
    "models/classification_model.pkl"
)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_student(
    study_hours,
    attendance,
    previous_marks,
    assignment_score,
    internal_marks,
    sleep_hours,
    internet_usage,
    extracurricular
):

    data = pd.DataFrame([{

        "study_hours": study_hours,

        "attendance": attendance,

        "previous_marks": previous_marks,

        "assignment_score": assignment_score,

        "internal_marks": internal_marks,

        "sleep_hours": sleep_hours,

        "internet_usage": internet_usage,

        "extracurricular": extracurricular

    }])


    # Predict final marks
    predicted_marks = regression_model.predict(
        data
    )[0]


    # Predict performance category
    predicted_category = classification_model.predict(
        data
    )[0]


    return predicted_marks, predicted_category


# ==========================================
# TEST STUDENT
# ==========================================

if __name__ == "__main__":

    print("\n====================================")
    print(" STUDENT PERFORMANCE PREDICTION")
    print("====================================")


    # Example student

    study_hours = 6

    attendance = 85

    previous_marks = 72

    assignment_score = 80

    internal_marks = 75

    sleep_hours = 7

    internet_usage = 3

    extracurricular = 1


    marks, category = predict_student(

        study_hours,

        attendance,

        previous_marks,

        assignment_score,

        internal_marks,

        sleep_hours,

        internet_usage,

        extracurricular

    )


    print("\nStudent Details")

    print(
        "Study Hours:",
        study_hours
    )

    print(
        "Attendance:",
        attendance,
        "%"
    )

    print(
        "Previous Marks:",
        previous_marks
    )

    print(
        "Assignment Score:",
        assignment_score
    )

    print(
        "Internal Marks:",
        internal_marks
    )

    print(
        "Sleep Hours:",
        sleep_hours
    )

    print(
        "Internet Usage:",
        internet_usage
    )

    print(
        "Extracurricular:",
        "Yes" if extracurricular == 1 else "No"
    )


    print("\n====================================")

    print(
        "Predicted Final Marks:",
        round(marks, 2),
        "/ 100"
    )

    print(
        "Performance Category:",
        category
    )

    print("====================================")
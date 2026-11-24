"""
Data Preprocessing Using NumPy, Pandas and Seaborn
Dataset: 20 students (Student Name, Age, Gender, Study Hours, Attendance, Marks)

Install once if needed:  pip install numpy pandas seaborn matplotlib
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# DATASET (20 rows). On purpose it contains:
#   - 1 duplicate record (Hina Malik appears twice)
#   - 3 missing values (None) so that Question 3 has something to handle
# ---------------------------------------------------------------------
data = {
    "Student Name": [
        "Ali Khan", "Sara Ahmed", "Bilal Raza", "Hina Malik", "Zain Ali",
        "Ayesha Noor", "Usman Tariq", "Fatima Zahra", "Hamza Sheikh", "Maryam Iqbal",
        "Hassan Raza", "Noor Fatima", "Omar Farooq", "Zara Khan", "Daniyal Shah",
        "Iqra Aslam", "Faizan Butt", "Sana Mirza", "Talha Javed", "Hina Malik",
    ],
    "Age": [20, 21, 19, 22, 20, 21, 23, 19, 20, 22, 21, 20, 24, 19, 21, 22, 20, 23, 21, 22],
    "Gender": [
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female",
        "Male", None, "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female",
    ],
    "Study Hours": [
        6.5, 8.0, 3.0, 7.5, 4.0,
        9.0, 2.5, 8.5, 5.0, 7.0,
        3.5, 6.0, 1.5, 8.0, None,
        7.2, 4.5, 6.8, 5.5, 7.5,
    ],
    "Attendance": [
        88, 95, 70, 92, 75,
        98, 60, 96, 80, 90,
        72, 85, 55, None, 82,
        91, 78, 89, 84, 92,
    ],
    "Marks": [
        72, 88, 45, 81, 55,
        93, 38, 90, 62, 78,
        50, 70, 35, 85, 68,
        80, 58, 76, 66, 81,
    ],
}

# ---------------------------------------------------------------------
# Q1. NumPy array of marks + mean, median, standard deviation
# WHY: NumPy arrays store numbers compactly and give fast math functions.
# ---------------------------------------------------------------------
print("=" * 60)
print("Q1. NumPy statistics of marks")
print("=" * 60)

marks_array = np.array(data["Marks"])
print("Marks array :", marks_array)
print("Mean        :", np.mean(marks_array))
print("Median      :", np.median(marks_array))
print("Std. dev.   :", round(np.std(marks_array), 2))

# ---------------------------------------------------------------------
# Q2. Create a Pandas DataFrame
# WHY: a DataFrame is a table (rows and columns) that is easy to clean,
#      filter and plot.
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("Q2. Pandas DataFrame")
print("=" * 60)

df = pd.DataFrame(data)
print(df)
print("\nShape (rows, columns):", df.shape)

# ---------------------------------------------------------------------
# Q3. Identify and handle missing values
# WHY: missing values (NaN) break calculations and charts, so we first
#      find them and then fill or remove them.
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("Q3. Missing values")
print("=" * 60)

print("Missing values per column:")
print(df.isnull().sum())

# Numeric columns -> fill with the column mean (a typical value)
df["Study Hours"] = df["Study Hours"].fillna(df["Study Hours"].mean())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
# Text column -> fill with the most common value (mode)
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())

# ---------------------------------------------------------------------
# Q4. Find duplicate records and remove them
# WHY: duplicates count the same student twice and distort the results.
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("Q4. Duplicate records")
print("=" * 60)

print("Number of duplicate rows:", df.duplicated().sum())
print("Duplicate row(s):")
print(df[df.duplicated()])

df = df.drop_duplicates().reset_index(drop=True)
print("\nShape after removing duplicates:", df.shape)

# ---------------------------------------------------------------------
# Q5. Filter students who scored more than 70 marks
# WHY: a boolean condition inside df[...] keeps only matching rows.
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("Q5. Students with marks above 70")
print("=" * 60)

top_students = df[df["Marks"] > 70]
print(top_students)
print("Count:", len(top_students))

# ---------------------------------------------------------------------
# Q6. Seaborn: relationship between study hours and marks
# WHY: a scatter plot shows whether more study hours go with higher marks.
#      regplot adds a trend line to make the pattern clear.
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="Study Hours", y="Marks", color="steelblue")
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours (per day)")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("study_hours_vs_marks.png", dpi=150)
plt.show()

print("\nCorrelation between Study Hours and Marks:",
      round(df["Study Hours"].corr(df["Marks"]), 2))

# ---------------------------------------------------------------------
# Q7. Histogram of student marks
# WHY: a histogram groups marks into ranges (bins) to show how the marks
#      are distributed (where most students fall).
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Marks", bins=8, kde=True, color="seagreen")
plt.title("Distribution of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("marks_histogram.png", dpi=150)
plt.show()

# Save the cleaned data as a file too
df.to_csv("students_clean.csv", index=False)
print("\nCleaned dataset saved to students_clean.csv")

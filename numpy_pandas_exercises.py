import numpy as np
import pandas as pd

# Exercise 1

scores = np.array([
    [78, 85, 90, 88],
    [92, 81, 76, 95],
    [89, 90, 91, 87],
    [65, 70, 72, 68],
    [99, 95, 98, 100]
])

# 1. Calculate the average score of each student
student_averages = scores.mean(axis=1)
print("\n***Exercise 1 - 1***")
print("Student averages:")
print(student_averages)

# 2. Calculate the average score of each exam
exam_averages = scores.mean(axis=0)
print("\n***Exercise 1 - 2***")
print("\nExam averages:")
print(exam_averages)

# 3. Subtract each exam average using broadcasting
adjusted_scores = scores - exam_averages
print("\n***Exercise 1 - 3***")
print("\nAdjusted scores:")
print(adjusted_scores)

# 4. Find the student with the highest average score
best_student = np.argmax(student_averages) + 1
print("\n***Exercise 1 - 4***")
print("\nStudent with the highest average:", best_student)

# Exercise 2

temps = np.array([22, 25, 19, 30, 28, 31, 24, 18, 35, 27], dtype=float)

# 1. Create a boolean mask for temperatures above 28
mask = temps > 28
print("\n***Exercise 2 - 1***")
print("\nTemperatures above 28 mask:")
print(mask)

# 2. Extract temperatures above 28
high_temps = temps[mask]
print("\n***Exercise 2 - 2***")
print("\nTemperatures above 28:")
print(high_temps)

# 3. Replace temperatures below 20 with NaN
temps[temps < 20] = np.nan
print("\n***Exercise 2 - 3***")
print("\nTemperatures after replacing values below 20:")
print(temps)

# 4. Calculate the average without considering NaN values
average_temp = np.nanmean(temps)
print("\n***Exercise 2 - 4***")
print("\nAverage temperature:")
print(average_temp)


# Exercise 3

data = [
    ("Alice", 34, 70000),
    ("Bob", 45, 85000),
    ("Charlie", 25, 50000),
    ("Diana", 40, 90000),
    ("Eve", 29, 62000)
]

# 1. Create a NumPy Structured Array
employees = np.array(
    data,
    dtype=[
        ("name", "U10"),
        ("age", "i4"),
        ("salary", "i4")
    ]
)

print("\n***Exercise 3 - 1***")
print("\nEmployees:")
print(employees)

# 2. Calculate the average salary of employees older than 30
average_salary_over_30 = employees[employees["age"] > 30]["salary"].mean()

print("\n***Exercise 3 - 2***")
print("\nAverage salary of employees over 30:")
print(average_salary_over_30)

# 3. Find employees whose salary is above the overall average
average_salary = employees["salary"].mean()
high_salary_employees = employees[employees["salary"] > average_salary]["name"]

print("\n***Exercise 3 - 3***")
print("\nEmployees above average salary:")
print(high_salary_employees)

# 4. Sort the array by salary in descending order
sorted_employees = np.sort(employees, order="salary")[::-1]

print("\n***Exercise 3 - 4***")
print("\nEmployees sorted by salary:")
print(sorted_employees)

# Exercise 4

data = {
    "product": ["A", "B", "C", "A", "B", "C"],
    "store": ["X", "X", "X", "Y", "Y", "Y"],
    "sales": [120, 150, 90, 200, 130, 160]
}

df = pd.DataFrame(data)

# 1. Calculate total sales for each product
product_sales = df.groupby("product")["sales"].sum()

print("\n***Exercise 4 - 1***")
print("\nTotal sales for each product:")
print(product_sales)

# 2. Calculate average sales for each store
store_sales = df.groupby("store")["sales"].mean()

print("\n***Exercise 4 - 2***")
print("\nAverage sales for each store:")
print(store_sales)

# 3. Select rows where sales are greater than 140
high_sales = df[df["sales"] > 140]

print("\n***Exercise 4 - 3***")
print("\nSales greater than 140:")
print(high_sales)

# 4. Create the normalized_sales column
df["sales_normalized"] = df["sales"] / df["sales"].max()

print("\n***Exercise 4 - 4***")
print("\nDataFrame with normalized sales:")
print(df)

# Exercise 5

data = {
    "student": ["A", "B", "C", "D", "E"],
    "math": [90, 85, None, 70, 88],
    "physics": [None, 80, 75, 65, 92],
    "chemistry": [85, None, 82, 60, None]
}

df = pd.DataFrame(data)

# 1. Count missing values in each column
missing_values = df.isna().sum()

print("\n***Exercise 5 - 1***")
print("\nMissing values:")
print(missing_values)

# 2. Replace missing values with the mean of each column
df["math"] = df["math"].fillna(df["math"].mean())
df["physics"] = df["physics"].fillna(df["physics"].mean())
df["chemistry"] = df["chemistry"].fillna(df["chemistry"].mean())

print("\n***Exercise 5 - 2***")
print("\nData after filling missing values:")
print(df)

# 3. Calculate the average score of each student
df["average"] = df[["math", "physics", "chemistry"]].mean(axis=1)

print("\n***Exercise 5 - 3***")
print("\nStudent averages:")
print(df[["student", "average"]])

# 4. Find the student with the highest average
best_student = df.loc[df["average"].idxmax(), "student"]

print("\n***Exercise 5 - 4***")
print("\nStudent with the highest average:")
print(best_student)

# Exercise 6

sales = np.array([
    [120, 135, 150, 160, 145, 170, 180],
    [130, 140, 155, 165, 150, 175, 190]
])

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# 1. Convert the array to a Pandas DataFrame
df = pd.DataFrame(
    sales,
    index=["Week1", "Week2"],
    columns=days
)

print("\n***Exercise 6 - 1***")
print("\nSales DataFrame:")
print(df)

# 2. Calculate total sales for each week
weekly_sales = df.sum(axis=1)

print("\n***Exercise 6 - 2***")
print("\nTotal sales for each week:")
print(weekly_sales)

# 3. Calculate total sales for each day
daily_sales = df.sum(axis=0)

print("\n***Exercise 6 - 3***")
print("\nTotal sales for each day:")
print(daily_sales)

# 4. Find the day with the highest total sales
max_sales = df.max().max()
position = df.stack().idxmax()

print("\n***Exercise 6 - 4***")
print("\nDay with the highest sales:")
print(f"{position[0]} - {position[1]}: {max_sales}")

# 5. Find days with sales above the overall average
overall_average = df.values.mean()
days_above_average = df[df > overall_average].stack().dropna()

# print(f"\nOverall average sales: {overall_average:.2f}")
print("\n***Exercise 6 - 5***")
print("\nDays with sales above the overall average:")
print(f"Overall average sales: {overall_average:.2f}", "\n\n", days_above_average)
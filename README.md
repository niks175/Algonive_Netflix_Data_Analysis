# Netflix Data Analysis and Visualization 📊

## Project Overview

This project focuses on analyzing Netflix Movies and TV Shows data to understand viewing trends, content distribution, and user preferences. The analysis was performed using Python in Google Colab with data visualization techniques to extract meaningful insights.

The goal of this project is to demonstrate data analytics skills including data cleaning, exploratory data analysis (EDA), feature engineering, clustering, and visualization.

---

## Dataset

The dataset used in this project contains information about Netflix movies and TV shows such as:

* Title
* Type (Movie or TV Show)
* Director
* Cast
* Country
* Release Year
* Date Added
* Rating
* Duration
* Genre

The dataset was sourced from Kaggle.

---

## Tools and Technologies

The following tools and libraries were used:

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Workflow

### 1. Data Collection

The Netflix dataset was downloaded from Kaggle and uploaded to Google Colab for analysis.

### 2. Data Cleaning

* Handled missing values
* Removed unnecessary null rows
* Converted date columns into proper datetime format

### 3. Exploratory Data Analysis (EDA)

Several analyses were performed including:

* Distribution of Movies vs TV Shows
* Top countries producing Netflix content
* Content added over the years
* Popular genres on Netflix

### 4. Feature Engineering

New features such as year_added and duration_numeric were created to enhance the analysis.

### 5. Clustering

A K-Means clustering algorithm was applied to group content based on features like release year and type.

### 6. Data Visualization

Visualizations were created to better understand patterns and trends in the dataset.

---

## Key Insights

* Netflix contains more movies than TV shows.
* The United States produces the highest amount of Netflix content.
* Content addition increased significantly after 2015.
* Certain genres dominate the platform.

---

## Project Structure

```
Netflix-Data-Analysis
│
├── netflix_titles.csv
├── Netflix_Data_Analysis.ipynb
├── cleaned_netflix_data.csv
└── README.md
```

---

## Results

This project demonstrates how data analytics can help understand content trends on streaming platforms and support better decision-making through data-driven insights.

---

## Author

Nikita

---

## Internship

This project was completed as part of a Data Analytics Internship.

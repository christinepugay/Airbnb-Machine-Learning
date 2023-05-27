# Airbnb Price Prediction Model using Machine Learning
As an Airbnb host, are you pricing your listings too high? or maybe too low? Having too high of a price maybe steering customers away from your listing and having too low of a price is probably resulting your business to lose profit.

In this project, we are analyzing both quantitative (such as number of beds, number of baths, etc.) and qualitative data (such as listing name, listing reviews, listing description, and host about) from Airbnb listings in Los Angeles Metropolitan Area to help optimize and build a price prediction machine learning model for Airbnb hosts. 


## About the dataset:  

Airbnb listings data was derived from Airbnb.com during the month of September 2022 in Los Angeles Metropolitan Area initially consisting 45,815 records and 74 features.  
## Exploratory Data Analysis
To get an overview of our dataset, we performed our initial exploratory data analysis. We made aware of null values, outliers, and other note-worthy insights from the dataset. 

## Text Preprocessing
As part of our analysis, I was assigned to perform text analysis to determine the correlation of the name, listing descriptions, host about, neighborhood overview, and listing reviews to the price. In [this notebook](https://github.com/christinepugay/Airbnb-Machine-Learning/blob/main/Text_Preprocessing.ipynb), you will see how I pre-processed these text data for the analysis. This includes counting words, sentences, removing stopwords from the texts. At the end, we are left with a csv file that will be used for our text analysis.

## Text Visualization Using WordCloud
After preprocessing, we were interested in the most common words used in these texts. We utilized word clouds to visualize the frequent and relevant word in these texts.

![image](https://github.com/christinepugay/Airbnb-Machine-Learning/assets/116247106/4e6e355d-3cff-408a-818a-379123dbb734)
![image](https://github.com/christinepugay/Airbnb-Machine-Learning/assets/116247106/73cbf12f-ae69-4a09-b5d4-3a830f52c735)


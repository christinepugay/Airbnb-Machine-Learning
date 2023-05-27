# Airbnb Price Prediction Model using Machine Learning
As an Airbnb host, are you pricing your listings too high? or maybe too low? Having too high of a price maybe steering customers away from your listing and having too low of a price is probably resulting your business to lose profit.

In this project, we are analyzing both quantitative (such as number of beds, number of baths, etc.) and qualitative data (such as listing name, listing reviews, listing description, and host about) from Airbnb listings in Los Angeles Metropolitan Area to help optimize and build a price prediction machine learning model for Airbnb hosts. 


## About the dataset:  

Airbnb listings data was derived from Airbnb.com during the month of September 2022 in Los Angeles Metropolitan Area initially consisting 45,815 records and 74 features.  
## Exploratory Data Analysis
To get an overview of our dataset, we performed our initial exploratory data analysis. We made aware of null values, outliers, and other note-worthy insights from the dataset. 

## [Text Preprocessing](https://github.com/christinepugay/Airbnb-Machine-Learning/blob/main/Text_Preprocessing.ipynb)
As part of our analysis, I was assigned to perform text analysis to determine the correlation of the name, listing descriptions, host about, neighborhood overview, and listing reviews to the price. In [this notebook](https://github.com/christinepugay/Airbnb-Machine-Learning/blob/main/Text_Preprocessing.ipynb), you will see how I pre-processed these text data for the analysis. This includes counting words, sentences, removing stopwords from the texts. At the end, we are left with a csv file that will be used for our text analysis.

## Text Visualization Using WordCloud
After preprocessing, we were interested in the most common words used in these texts. We utilized word clouds to visualize the frequent and relevant word in these texts. To provide more context to our word clouds, we decided to bin the listings into two categories: low-price and high-price listings. This will allow us to see a clear distinction on how hosts of low and high priced listings market their listings using the text discriptions on their page. 

Let's take a look at the frequently used words in Aibnb listings' Name!

<p align= "center"

**Low-Priced Listings with Median Price of $90**
![image](https://github.com/christinepugay/Airbnb-Machine-Learning/assets/116247106/4e6e355d-3cff-408a-818a-379123dbb734)
   
</p>

<p align = "center"
   
**High-Priced Listings with Median Price of $1400**
![image](https://github.com/christinepugay/Airbnb-Machine-Learning/assets/116247106/73cbf12f-ae69-4a09-b5d4-3a830f52c735)

</p>

Low-priced listings tend to have words such as private, home, and studio, describing the type of property that they offer. They also tend to describe their listings using adjectives such as cozy, beautiful, lovely, spacious, and quiet. In the high-priced listings, hosts tend to showcase the location of their listings. We can see that popular cities in Los Angeles are prominent in the word cloud such as: malibu, beverly, and venice. They also associate their listings with keywords such as luxury, oasis, spa, and retreat. In both of these word clouds, the word "hollywood" is prominent, we are assuming that majority of the listings are located in the Hollywood area. 

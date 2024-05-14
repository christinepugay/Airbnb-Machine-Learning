# Airbnb Price Prediction Model using Machine Learning

In this project, we are analyzing both quantitative (such as number of beds, number of baths, etc.) and qualitative data (such as listing name, listing reviews, listing description, and host about) from Airbnb listings in Los Angeles Metropolitan Area to help optimize and build a price prediction machine learning model for Airbnb hosts. 


## About the dataset:  

Airbnb listings data was derived from Airbnb.com during the month of September 2022 in Los Angeles Metropolitan Area initially consisting 45,815 records and 74 features.  
## Exploratory Data Analysis
To get an overview of our dataset, we performed our initial exploratory data analysis. We made aware of null values, outliers, and other note-worthy insights from the dataset. 

## [Text Preprocessing](https://github.com/christinepugay/Airbnb-Machine-Learning/blob/main/Text_Preprocessing.ipynb)
As part of our analysis, I was assigned to perform text analysis to determine the correlation of the name, listing descriptions, host about, neighborhood overview, and listing reviews to the price. In [this notebook](https://github.com/christinepugay/Airbnb-Machine-Learning/blob/main/Text_Preprocessing.ipynb), you will see how I pre-processed these text data for the analysis. This includes counting words, sentences, removing stopwords from the texts. At the end, we are left with a csv file that will be used for our text analysis.

## [Text Visualization Using WordCloud](https://github.com/christinepugay/Airbnb-Machine-Learning/blob/main/Christine_WordCloud.ipynb)
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

## [Text Analysis Using Linear Regression](https://github.com/christinepugay/Airbnb-Machine-Learning/blob/main/Christine_POS_analysis.ipynb)
As part of our text analysis, we were interested in the relationship of the specific part of speech (POS) counts (such as nouns, adjectives, and verb count), word count, and sentence count to the listing price.  Here we performed linear regression using different numbers of predictors to analyze this relationship.

We took all text columns (name, description, neighborhood overview, and host about) and counted specific POS counts, word count, and sentence count. It's important to note that this procedure was performed on the original text, after HTML tags were removed. We were able to retrieve 21 different features including the listing price and 45,214 records.

**Pearson Correlation**

As part of our feature selection, we also performed a Pearson correlation. This will help us identify the relationship between the price and the features. Below we can see the correlation ratio of each of the features to the price in descending order. We will be using the top 10 and top 12 features from this table as two sets of predictors in our model. 

Here lists the correlation of each features to the price in a descending order:

price	1.000000

noun_count_description	0.106512

noun_count_name	0.090874

name_TC	0.087481

description_TC	0.075893

adj_count_description	0.071835

verb_count_description	0.069159

name_SC	0.033488

verb_count_name	0.028551

noun_count_host_about	0.020103

verb_count_host_about	0.015920

host_about_TC	0.011709

verb_count_neighborhood_overview	0.011477

description_SC	0.006390

adj_count_host_about	-0.001212

noun_count_neighborhood_overview	-0.001388

neighborhood_overview_TC	-0.002529

host_about_SC	-0.002882

neighborhood_overview_SC	-0.012026

adj_count_neighborhood_overview	-0.017217

adj_count_name	-0.039824

**Linear Regression Model**

Linear regression will be using as our model to predict our target variable, price. Linear regression is a simple model that can easily be implemented to our problem. We are also assuming that there is a linear relationship between the predictors (POS, word, and sentence count) and our outcome, price. 

We are using 80% of the data to train the model while the remaining 20% will be used as our validation set.

We started off by using all 20 features as our predictor and this resulted in 0.046 r2, 63873.14 MSE, and finally 252.73 RMSE. Looking at the r2, the result is too small indicating that the model isn't able to explain a lot of the variation in our data. We then moved on to manually pick 8 features and 12 features as our predictors. These also resulted in less than 3% r2 score. The same pattern of significantly small r2 score can be seen when we used the top 10 and top 12 features using Pearson correlation.

**Results**

20 features: All Features 

Feature Selection Method: Manual

      R2 score: 0.046
      
		MSE: 63873.14
      
		RMSE: 252.73
      
8 features: Token and Sentence Count	

Feature Selection Method: Manual	

      R2 score: 0.02
      
		MSE: 66419.86
      
		RMSE: 257.72
      
12 features: noun, adjective, and verb count	

Feature Selection: Manual	

      R2 score: 0.023
      
		MSE: 65303.38
      
		RMSE: 255.54
      
Top 10 Pearson correlation results 	

Feature Selection: Pearson Correlation	

      R2 score: 0.024
      
		MSE: 63442.50
      
		RMSE: 251.88
      
Top 12 Pearson Correlation (positive correlation to price)	

Feature Selection: Pearson Correlation	

      R2 score: 0.032
      
		MSE: 65101.52
      
		RMSE: 255.15
      

Considering this result, we can conclude that there isn't any significant relationship between the counts of POS (nouns, adjectives, and verb count), word, and sentences in the text components (name, description, neighborhood overview, and host about) of an Airbnb listing to its price.

# Top Features for Machine Learning Models using Pearson Correlation

To assist my team with identifying the overall top correlated features for their machine learning models, I also used pearson correlation to identify the top features that correlates with the price. Below is the visual represation of the results:

![](https://github.com/christinepugay/Airbnb-Machine-Learning/blob/main/correlation_heatmap.png)












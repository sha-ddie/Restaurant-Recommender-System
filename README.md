## Dining Guide: Personalized Restaurant Recommendations

## Group Members:
1. Wendy Muturi
2. Mulei Mutuku
3. Margaret Mitey
4. Jeff Kiarie
5. Linus Gichuhi
6. Joshua Ooko

![Restaurant](https://www.falstaff-travel.com/wp-content/uploads/2022/07/Sea-view-seafood-greek-dishes-landscape-table-chairs-cycladic-decoration-aegean-colours-blue-horizon-summer-Medusa-restaurant-Milos-island-Greece-1-1024x750.jpg)

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Main Objective](#main-objective)
- [Specific Objective](#specific-objective)
- [Data understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Explatory Data Analysis](#explatory-data-analysis)
- [Modelling](#modelling)
- [Content-based Recommendation system](#content-based-recommendation-system)
- [Collaborative filtering](#collaborative-filtering)
- [Evaluation](#evaluation)
- [Minimum Viable Product](#minimum-viable-product)
- [Conclusions](#conclusions)
- [Recommendation](#recommendation)
- [Future Improvement Ideas](#future-improvement-ideas)


# Introduction
In an age where culinary diversity and dining out have become integral parts of our social fabric, choosing the perfect restaurant can be both exciting and overwhelming. With an abundance of dining options ranging from quaint bistros to exotic eateries, making a dining decision has never been more challenging.
Traditional restaurant websites have long relied on filters based on amenities, location, or cuisine types, providing users with a plethora of options to sift through. However, as the restaurant industry evolves and culinary landscapes expand, the need for a more refined and personalized approach to restaurant discovery has become evident.
Enter the era of restaurant recommendation systems—a technological marvel that goes beyond the mundane task of filtering restaurants based on their amenities. These systems leverage the power of data science, machine learning, and user preferences to deliver tailored dining suggestions that match your unique tastes and preferences.

In a world where time is precious and choices are abundant, restaurant recommendation systems offer an invaluable solution by enhancing the dining experience in ways that traditional filters simply cannot.
This project delves into the world of restaurant recommendation systems, exploring their importance, functionality, and the transformative impact they have on the way we discover and enjoy culinary delights.
We will unveil how these intelligent algorithms are reshaping the gastronomic landscape, catering to the ever-evolving preferences of diners and revolutionizing the art of restaurant selection. Join us on this journey as we unravel the magic of restaurant recommendation systems, offering a taste of the future of dining exploration.

# Problem Statement
This project aims to address the challenge faced by individuals in making informed choices about restaurants and dining experiences by developing a user-friendly restaurant recommendation system that empowers individuals to make informed dining decisions, ultimately enhancing their overall restaurant experience.

# Main Objective

To develop an interactive and user-friendly restaurant recommendation system.

# Specific Objective

a) Analyze key factors for restaurant ratings, identifying and evaluating the key attributes and factors that significantly influence restaurant ratings and customer preferences using data analysis techniques. 

b) Develop content-based recommendation algorithms, creating and implementing advanced content-based on algorithms that can generate personalized restaurant recommendations based on user-defined text, restaurant names, and other user preferences. 

c) Integrate interactive maps to create an interactive mapping feature within the recommendation system. This map will allow users to explore geographic trends in restaurant recommendations, providing a visually engaging way to discover dining options based on location.

d) Build an interactive user interface that allows users to easily access and interact with the restaurant recommendation system.

# Data understanding

The dataset used in this project, was extracted from the Yelp Restaurant [database](https://www.yelp.com/dataset), which is publicly available and contains a large number of reviews across various restaurants and locations. The dataset contains 908,915 tips/reviews by 1,987,897 users on the  131,930 businesses and their attributes like hours, parking, availability, and ambience aggregated check-ins over time for each. The **dataset contains five jason files namely business.json, checkin.json, review.json, tips.json and user.json**, but only two files were found to containe the relevant required information;

> **business.json**: this json file has data on various business all spread over different US states and their relevant attributes.

> **review.json**: this json file contains information on reviews made by different users on various business they were served.

Due to the dataset being large we have only extracted 54,380 rows and 14 columns which are enough for our analysis and the two above stated json files were merged and only the relevant columns were maintained, named;
- **user_id:** A unique identifier for each user who submitted a review

- **business_id:** A unique identifier for each business being reviewed

- **name:** string, the business's name

- **address:** string, the full address of the business 

- **stars:** The rating given by the user in terms of stars (e.g., 1.0, 2.0, 3.0, 4.0, 5.0),

- **text:** The actual text content of the review and

- **review_count:** number of reviews the business has received

- **city:** string, the city eg "San Francisco",
 
- **state:** string, 2 character state code, if applicable eg"CA",

- **latitude:**  float, latitude of the business

- **longitude:** float, longitude of the business

- **attributes:** business attributes and features

- **categories:** a list of the business categories

- **hours:** hours in when the business is open,hours are using a 24hr clock

For download of the dataset's, view the [Link](https://www.yelp.com/dataset) anf for complete [documentation](https://www.yelp.com/dataset/documentation/main) of all the datasets.

# Data Preparation

In this section, we will perform data cleaning to prepare the dataset for analysis, the various data cleaning methods that are to be used will be;

- Renaming columns
- Checking Dealing with missing data
- Checking and removing duplicates 
- Feature Engineering
- Selecting the Relevant Columns
- Droping Irrelevant columns
- Selecting relevant rows

# Explatory Data Analysis

Conducting a thorough exploratory data analysis (EDA) is pivotal in crafting an interactive and user-friendly restaurant recommendation system. The analysis delved into critical dataset features, examining the distribution of ratings, categories, and restaurants across cities and states, as well as popular restaurants.
Visualizations, including histograms, box plots, and hexbin plots, were employed for a comprehensive understanding. Histograms shed light on the distribution of ratings, categories, and restaurant counts in cities and states, while box plots showcased business ratings against the price ranges. The hexbin plot illustrated the relationship between business ratings and the number of reviews. Word clouds were generated to highlight common words in positive and negative reviews, and a map visualized restaurant locations.
This EDA offered vital insights, identifying key dataset features.
The dataset is primarily composed of food-related establishments, highlighting a diverse range of cuisines. Nightlife venues are also notable, indicating a vibrant nightlife scene. Conversely, fast food and burger establishments are less prevalent in the dataset.
![Alt text](image.png)

Philadelphia stands out as the most prevalent city in the dataset, boasting the highest number of restaurants. Tampa follows, though not as closely, indicating a noteworthy restaurant presence. In contrast, Reno and Santa Barbara have fewer restaurants, making them less common in this dataset.
![Alt text](image-1.png)

The review counts for the restaurants are listed in descending order, with Luke leading at 4554 reviews, followed by Santa Barbara Shellfish Company with 2404 reviews, Cafe Fleur De Lis with 1865 reviews, and Milk and Honey Nashville with 1725 reviews. These counts reflect the popularity and customer engagement of each restaurant.
![Alt text](image-2.png)

# Modelling

Prior to developing recommendation system models, sentiment analysis was done after conducting essential text preprocessing steps to enhance the dataset for analysis. This includes feature engineering, involving the creation of a new "review" column that consolidates all text reviews for a specific restaurant. Additionally, we will execute procedures such as punctuation removal, stopword elimination using the RegexpTokenizer() method, stemming for word simplification with the SnowballStemmer() method, and word vectorization through the TfidfVectorizer() method. These processes collectively contribute to optimizing the dataset for subsequent natural language processing (NLP) analysis.
After initializing the tokenizer and stemmer, we proceed to compute text frequency-inverse document frequency (TF-IDF) values using the TfidfVectorizer() method. This step is pivotal in transforming unstructured text into a structured numerical format suitable for diverse natural language processing (NLP) and text mining tasks, facilitating analysis and machine learning applications. The code configures and employs the TF-IDF vectorizer to convert text data into a numerical representation, capturing word importance and ensuring uniformity by stemming stopwords. Subsequently, the code calculates cosine similarity between rows of the TF-IDF matrix, measuring the similarity in 'details' text descriptions among different businesses based on their TF-IDF scores.
We will pickle our desired data for deployment.


# Content-based Recommendation system

Using the cosine similarity matrix, our content-based recommendation system suggests restaurants to users based on the similarity between restaurant names or specified attributes. This involves comparing user preferences with various restaurants and recommending the top similar options to cater to individual tastes.

# Collaborative filtering

In building a collaborative filtering recommendation system with the Surprise library, we selected relevant columns and initialized a Reader object to format the data. Subsequently, we loaded the data into a Surprise Dataset for further analysis and model creation. We then compared various neighborhood-based models to identify the top performer based on the RMSE metric. Following this, we compared the neighborhood-based model with model-based models to determine the best overall model for our recommendation system. 
The following steps were taken:
Fisrtly , we model a baseline SVD() model using the default parameters. The first baseline model had an RMSE of 1.256 same as our best neighborhood based model which had an RMSE of 1.257. Using the GridSearchCv we will tune the SVD model inorder to improve the training RMSE scores.
The SVD collaborative filtering model undergoes hyperparameter tuning through grid search and cross-validation. The optimized model achieves an RMSE of approximately 1.25, signifying good predictive accuracy. The MAE value is around 1.01, indicating improved prediction accuracy. The best hyperparameters include 'n_factors' = 20 and 'reg_all' = 0.05 for RMSE, and 'n_factors' = 20 and 'reg_all' = 0.02 for MAE. These settings make the SVD model well-suited for personalized recommendations based on user ratings.
Finally, the code created initiates an SVD model with tailored hyperparameters, training it on the dataset for personalized user recommendations. To tackle the cold start issue, a function named **restaurant_rater()** engages users to input ratings for specific restaurants. This data is collected for analysis or to support the recommendation system. In scenarios where no user ratings exist, the function seamlessly transitions to the content-based system, effectively addressing the cold start problem.
Link to the analysis [notebook](Phase-5-Capstone-Project/README.md Phase-5-Capstone-Project/student.ipynb)

# Evaluation

Effectively addressing the "cold start problem" is crucial for our model, ensuring meaningful recommendations for new users or restaurants with limited review data. Geographical coverage expansion is also a key metric, with success defined by the model providing relevant recommendations for users across various regions and cities. The successful deployment of our recommendation model is a critical evaluation metric, emphasizing accessibility, responsiveness, and the ability to generate real-time recommendations.

# Minimum Viable Product

The rated() function lays the foundation for a collaborative filtering function utilizing the SVD model. When encountering a user without any entered ratings for suggested restaurants (addressing the cold start problem), the function seamlessly switches to recommending using the content-based system, thus ensuring a minimum viable product with a versatile recommendation approach.

# Conclusions

In conclusion, this project has successfully achieved its main objective of developing an interactive and user-friendly restaurant recommendation system. This system not only provides personalized dining suggestions but also takes into account various factors that influence restaurant ratings and user preferences. The integration of an advanced recommendation algorithm ensures that users can access tailored restaurant recommendations, enhancing their overall dining experiences.

Throughout this project, we also met specific objectives. We designed and developed a user-friendly website, making it convenient for users to interact with our recommendation system. Additionally, we conducted in-depth analyses of the factors that significantly impact restaurant ratings and user preferences. This understanding was crucial in refining our recommendation algorithms, ensuring they provide valuable and relevant suggestions to users.

Furthermore, we harnessed the power of geographical data visualization using Folium. By creating interactive maps, we were able to explore geographic trends related to restaurant recommendations. These maps not only make our recommendations more engaging but also help users discover new dining experiences in their preferred locations.

In summary, this project's multifaceted approach aimed at delivering a holistic restaurant recommendation system has proven successful. Users can now access personalized dining recommendations, taking into account various influencing factors and geographical trends. This project not only achieves its specific objectives but also offers a valuable service that enhances the dining experiences of users.

# Recommendation

a) Integration of user feedback: Actively seek and integrate user feedback to refine and improve the recommendation system

b) Enhanced user profiles: Use this data to provide more personalized restaurant recommendations.

c) Enhance recommendation algorithms: Continue to refine and enhance the recommendation algorithms. Explore more advanced machine learning techniques, including deep learning, to improve recommendation accuracy and personalization.

d) Expand geographical coverage: Gradually expand the geographical coverage of the recommendation system to include more regions and cities, providing users with a broader range of dining options.

# Future Improvement Ideas

a) Enhanced visuals: Incorporate images and visual content, such as restaurant photos and dishes, to make the recommendations more visually appealing and informative.

b) Community and social sharing: Encourage users to share their dining experiences and reviews within the system. Implement social sharing features to build a user community and facilitate restaurant recommendations from peers.

c) Real-time updates: Enable real-time updates of restaurant information, including opening hours, special offers, and menu changes. Users should receive the most current and accurate data.

d) Integration with food delivery services: Collaborate with food delivery services to allow users to place food orders for delivery or pickup directly through the recommendation system.

e) Advanced machine learning algorithms: Explore the use of advanced machine learning algorithms to further enhance recommendation accuracy.

# Resources

1: For the complete analysis, here is the [Notebook](https://github.com/sha-ddie/Phase-5-Capstone-Project/blob/main/student.ipynb)

2: The presentation slide are in this [Link](https://www.canva.com/design/DAFxJG67E08/dLd8YHTDIMBAeTA48jvuDA/view?utm_content=DAFxJG67E08&utm_campaign=designshare&utm_medium=link&utm_source=editor)



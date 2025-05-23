# Recommendation System for Indonesia Tourism Destination - Fatih El Haq

## Project Overview

Indonesia's tourism sector plays a crucial role in its economy, contributing significantly to foreign exchange revenues and regional development (The Jakarta Post, 2018). However, the abundance of diverse destinations across the archipelago often overwhelms potential visitors, leading to decision fatigue. This challenge underscores the necessity for an effective recommendation system that can assist tourists in selecting destinations that align with their preferences and interests.

Recent studies have demonstrated the effectiveness of various recommendation algorithms in enhancing the tourism experience. For instance, a study on Bali's tourism sector employed a Weighted Hybrid method combining Collaborative Filtering and Content-Based approaches, yielding improved accuracy in destination suggestions (Pratama et al., 2023). Similarly, research focused on Madura Island utilized a Collaborative Filtering-based system with modified Cosine similarity and Convolutional Neural Networks, achieving a low RMSE of 0.2579, indicating precise personalized recommendations (Permana et al., 2024). 
The implementation of such systems is increasingly relevant in the context of e-tourism, which emphasizes the integration of advanced technologies like Artificial Intelligence and Big Data to enhance tourism experiences (Samara et al., 2020). By adopting these technologies, Indonesia can offer tailored travel suggestions that cater to individual preferences, thereby improving tourist satisfaction and supporting sustainable tourism development.

In conclusion, addressing the information overload faced by tourists through personalized recommendation systems is essential for optimizing travel experiences in Indonesia (Pratama et al., 2023). Leveraging advanced algorithms and technologies will not only assist tourists in making informed decisions but also contribute to the sustainable growth of the nation's tourism industry.

## Business Understanding

Indonesia's tourism sector offers a wide range of destinations, from natural wonders and cultural heritage sites to modern entertainment venues. Despite this diversity, many domestic and international tourists face difficulties in identifying destinations that align with their interests, largely due to the overwhelming number of options and the absence of personalized guidance. This situation often leads to inefficient travel planning and underutilization of tourism potential. Fortunately, available datasets include curated information on Indonesian tourism destinations, their categories, and user-generated ratings, offering an opportunity to develop a data-driven recommendation system.

### Problem Statements
To address the challenges in personalized destination discovery, this project defines the following problem statements:
- How can a personalized destination recommendation system be created using content-based filtering techniques based on destination data?
- How can other potentially preferred and unvisited destinations be recommended to users based on existing ratings data?

### Goals
To answer the problem statements, the project aims to achieve the following goals:
- Develop a personalized recommendation system that suggests destinations similar to user preferences using content-based filtering.
- Recommend destinations that match user preferences and have not yet been visited by leveraging collaborative filtering techniques.

### Solution Statements
To achieve these goals, the project applies two recommendation system approaches:
- **Content-Based Filtering**: Utilizes TF-IDF Vectorizer to transform destination text attributes into weighted numerical vectors and applies Cosine Similarity to calculate the similarity between these vectors, enabling the system to recommend destinations similar to those preferred by the user.
- **Collaborative Filtering**: Implements RecommenderNet, a deep learning model that captures latent features from user-place interaction data (ratings) to predict preferences for unrated destinations, thereby supporting personalized recommendations.

## Data Understanding

The dataset used in this project is sourced from the Indonesia Tourism Destination dataset available on Kaggle ([link](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination?select=tourism_with_id.csv)). The dataset comprises approximately 400 unique tourist attractions distributed across five major Indonesian cities, dummy user data with 300 users, and a ratings dataset containing 10,000 entries of user-place interactions. The data presents an opportunity to build a recommendation system by leveraging both descriptive features of destinations and user ratings.

### Variable Description
This dataset also consists of 3 variables, namely:
1. place: contains information on tourist attractions in 5 major cities in Indonesia totaling ~400
2. user: contains dummy user data to make recommendation features based on user
3. rating: contains 3 columns, namely the user, the place, and the rating given, serves to create a recommendation system based on the rating

### Exploratory Data Analysis
Exploratory analysis of the dataset reveals a rich combination of categorical, numerical, and textual data, which supports both content-based and collaborative filtering approaches. The geographical spread and categorical diversity also enable nuanced recommendation possibilities. Initial visualizations or statistical summaries provide insights into data distribution, missing values, and potential preprocessing needs to improve model performance.

**Table 1 Dataset Summary**
| DataFrame Name | Column Name    | dtype     | Minimum Value | Maximum Value | Mean Value   | Median Value | Standard Deviation | Number of Rows | Number of Missing Values | Number of Unique Values | Number of Duplicated Values |
|----------------|---------------|-----------|---------------|---------------|--------------|--------------|--------------------|----------------|--------------------------|------------------------|-----------------------------|
| place_df       | Place_Id      | int64     | 1.0           | 437.0         | 219.0        | 219.0        | 126.295289         | 437            | 0                        | 437                    | 0                           |
| place_df       | Place_Name    | object    | NaN           | NaN           | NaN          | NaN          | NaN                | 437            | 0                        | 437                    | 0                           |
| place_df       | Description   | object    | NaN           | NaN           | NaN          | NaN          | NaN                | 437            | 0                        | 437                    | 0                           |
| place_df       | Category     | object    | NaN           | NaN           | NaN          | NaN          | NaN                | 437            | 0                        | 6                      | 431                         |
| place_df       | City          | object    | NaN           | NaN           | NaN          | NaN          | NaN                | 437            | 0                        | 5                      | 432                         |
| place_df       | Price         | int64     | 0.0           | 900000.0      | 24652.173913 | 5000.0       | 66446.374709       | 437            | 0                        | 50                     | 387                         |
| place_df       | Rating        | float64   | 3.4           | 5.0           | 4.442792     | 4.5          | 0.208587           | 437            | 0                        | 14                     | 423                         |
| place_df       | Time_Minutes  | float64   | 10.0          | 360.0         | 82.609756    | 60.0         | 52.872339          | 437            | 232                      | 15                     | 421                         |
| place_df       | Coordinate    | object    | NaN           | NaN           | NaN          | NaN          | NaN                | 437            | 0                        | 437                    | 0                           |
| place_df       | Lat           | float64   | -8.197894     | 1.078880      | -7.095438    | -7.020524    | 0.727241           | 437            | 0                        | 437                    | 0                           |
| place_df       | Long          | float64   | 103.931398    | 112.821662    | 109.160142   | 110.237468   | 1.962848           | 437            | 0                        | 437                    | 0                           |
| place_df       | Unnamed: 11   | float64   | NaN           | NaN           | NaN          | NaN          | NaN                | 437            | 437                      | 0                      | 436                         |
| place_df       | Unnamed: 12   | int64     | 1.0           | 437.0         | 219.0        | 219.0        | 126.295289         | 437            | 0                        | 437                    | 0                           |
| user_df        | User_Id       | int64     | 1.0           | 300.0         | 150.5        | 150.5        | 86.746758          | 300            | 0                        | 300                    | 0                           |
| user_df        | Location      | object    | NaN           | NaN           | NaN          | NaN          | NaN                | 300            | 0                        | 28                     | 272                         |
| user_df        | Age           | int64     | 18.0          | 40.0          | 28.7         | 29.0         | 6.393716           | 300            | 0                        | 23                     | 277                         |
| user_df        | Age_Group     | category  | NaN           | NaN           | NaN          | NaN          | NaN                | 300            | 0                        | 4                      | 296                         |
| rating_df      | User_Id       | int64     | 1.0           | 300.0         | 151.2927     | 151.0        | 86.137374          | 10000          | 0                        | 300                    | 9700                        |
| rating_df      | Place_Id      | int64     | 1.0           | 437.0         | 219.4164     | 220.0        | 126.228335         | 10000          | 0                        | 437                    | 9563                        |
| rating_df      | Place_Ratings | int64     | 1.0           | 5.0           | 3.0665       | 3.0          | 1.379952           | 10000          | 0                        | 5                      | 9995                        |

Table 1 shows the summary for all table from the dataset. It shows the statistical summaries that consist of Min-Max values, mean, median, standard deviation, number of rows, number of missing values, unqiue values, and duplicated values. Furthermore, analysis for visualization of important features are below:

**Image 1 Destination (Place) by Category and City**
<div align="left">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/master/img/place_categories_cities.jpg" alt="img" height="100%">
</div>
<br>

The first graph in Image 1, "Count of Places by Category," details the categorical distribution of tourist destinations. "Taman Hiburan" (Amusement Park) represents the largest category with 135 attractions, closely followed by "Budaya" (Culture) with 117 attractions. "Cagar Alam" (Nature Reserve) comprises 106 attractions. Categories with fewer entries include "Bahari" (Marine) at 47, "Tempat Ibadah" (Places of Worship) at 17, and "Pusat Perbelanjaan" (Shopping Center) at 15. This categorical breakdown indicates a significant concentration of entertainment, cultural, and natural attractions within the dataset, with other categories being less prominent.

The second graph in Image 1, "Count of Places by City," illustrates the geographical distribution of these attractions across five major Indonesian cities. Yogyakarta and Bandung host the highest number of tourist destinations, with 126 and 124 attractions, respectively. Jakarta contains 84 attractions. Semarang and Surabaya have fewer attractions, with 57 and 46 respectively. This city-based analysis reveals that Yogyakarta and Bandung are primary locations for tourist destinations within this dataset, offering a broader range of attractions compared to Jakarta, Semarang, and Surabaya.

**Image 2 Destination (Place) by Price**
<div align="left">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/master/img/place_price.jpg" alt="img" width="50%">
</div>
<br>

Based on Image 2, the graph clearly shows a highly right-skewed distribution. A large majority of tourist attractions have an entry price close to 0 IDR, as evidenced by the tall bar at the lowest price range. As the price increases, the frequency of attractions significantly decreases, with very few places having high entry fees. This indicates that most tourist destinations in the dataset are either free or have very low admission costs, while a small number of attractions have considerably higher prices, aligning with the description of a skewed distribution for the `Price` variable.

**Image 3 User by Age**
<div align="left">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/master/img/user_age.jpg" alt="img" width="50%">
</div>
<br>

Based on Image 3,the largest age group is 18-24 years old, comprising 87 users. The 30-34 age group follows closely with 75 users, while the 25-29 age group has 74 users. The 35-40 age group represents the smallest segment, with 64 users. This distribution indicates a relatively even spread of users across the younger adult age ranges (18-34), with a slight decrease in the oldest age bracket (35-40). The data reflects that the majority of users in the dataset fall within the 18 to 34 age range, consistent with the reported mean user age of 28.7 years.

**Image 4 Rating Distribution**
<div align="left">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/master/img/rating_rating.jpg" alt="img" width="50%">
</div>
<br>

Based on Image 4, the distribution of ratings is relatively uniform across values 2, 3, 4, and 5, with each of these ratings having a frequency of over 2000. Rating 1 has a notably lower frequency, around 1700. This suggests that users generally provide positive feedback, as evidenced by the high counts for ratings 2 through 5. The mean rating of 3.07, as stated in the variable description, is consistent with this distribution, indicating a tendency for ratings to cluster around the middle to higher end of the scale.

To summarize, the dataset consists of three primary tables based Table 1 and all visualization:

1. **place_df** — contains detailed information about tourist destinations including:
   - `Place_Id`: Unique sequential identifier for each destination (1–437).
   - `Place_Name` and `Description`: Textual information describing each place.
   - `Category`: Six distinct categories such as 'Budaya', 'Taman Hiburan', and 'Bahari'.
   - `City`: Five cities where the destinations are located.
   - `Price`: Ticket or entry price ranging from 0 to 900,000 IDR, with a skewed distribution.
   - `Rating`: Average user rating between 3.4 and 5.0, showing generally positive feedback.
   - `Time_Minutes`: Estimated visiting duration, though with about 53% missing values.
   - `Lat` and `Long`: Geographical coordinates for mapping purposes.
   - Two unnamed columns, one of which is entirely missing and the other duplicating `Place_Id`.

2. **user_df** — provides user demographic information such as:
   - `User_Id`: Unique user identifier (1–300).
   - `Location`: User locations with 28 distinct values, some possibly inconsistent.
   - `Age`: User age ranging from 18 to 40 years, with a mean of 28.7.

3. **rating_df** — captures user feedback through:
   - `User_Id` and `Place_Id` references linking users to destinations.
   - `Place_Ratings`: Integer ratings from 1 to 5, with a mean rating of 3.07.

## Data Preparation
Data preparation ensures the dataset is clean, consistent, and well-structured, which is essential for accurate modeling and reliable recommendations. Each step targets a specific issue that could otherwise degrade model performance or cause bias.

### **Handling Missing Values:**  
Missing or irrelevant data can introduce noise and reduce model accuracy. Removing columns with excessive missing values or redundant data avoids introducing bias and complexity.  
- The `Unnamed: 11` column was dropped because it contained 100% missing values and no useful information.  
- The `Unnamed: 12` column was removed as it duplicated the `Place_Id` feature, preventing redundancy and potential confusion in joins.  
- The `Time_Minutes` column was dropped due to over 50% missing data, which could bias the dataset if imputed or filtered improperly.

### **Outlier Treatment:**  
Outliers can skew model training, especially for algorithms relying on distance or similarity metrics. Proper transformation and capping of extreme values stabilize variance and improve model robustness.  
- The `Price` feature was heavily right-skewed; applying a logarithmic transformation compressed extreme values to reduce variance and improve model convergence.  
- Winsorization of `Price` capped the extreme values, limiting their undue influence while preserving relative order.

### **Standardizing Categorical Data:**  
Inconsistent categorical values can fragment encoding and reduce model effectiveness. Standardizing ensures categories are meaningful and comparable.  
- Location names in `user_df` had inconsistent spellings and formats (e.g., “Jakarta” vs. “DKI Jakarta”). These were standardized by extracting the province name only, improving encoding consistency.

### **Removing Duplicate Entries:**  
Duplicate records can distort the distribution of user-item interactions, biasing recommendations. Ensuring unique user-place pairs maintains data integrity.  
- Duplicate `(User_Id, Place_Id)` entries in `rating_df` were removed to keep each interaction unique and avoid skewing the model.

### **Feature Enrichment via Joins:**  
Integrating user and place features with rating data enriches the dataset, enabling hybrid recommendation approaches that leverage both content and collaborative signals.  
- User and place attributes were joined into the rating dataset on `User_Id` and `Place_Id`, providing additional context for latent feature extraction in the recommendation model.

These data preparation steps collectively reduce noise, handle missing and extreme values, unify categorical features, and enrich the dataset, establishing a strong foundation for building an effective and accurate recommendation system.

## Modeling
### Model 1 - Content-Based Filtering

The first recommendation approach implemented is a content-based filtering model. This method focuses on recommending places similar to those a user has previously visited by leveraging place-specific attributes. The primary advantage of this approach is its ability to provide personalized recommendations without requiring extensive user interaction data. However, it may suffer from limited diversity and difficulty recommending completely new or less-characterized places.

The modeling process begins by preparing the feature data. A new combined feature, `Category_place__province`, is created by concatenating the place's category and province to better capture contextual similarities. This composite feature enables the model to consider both the type of attraction and its geographic location.

```python
content_df = place_prep.copy()
content_df['Category_place__province'] = content_df['Category'] + '__' + content_df['place_province']
``` 

A TF-IDF vectorizer is applied to this combined text feature, transforming categorical data into a numerical representation. This step calculates the importance of each category-province term relative to all places, resulting in a sparse matrix of shape (437, 25), where rows represent places and columns represent unique category-province tokens.

```python
tf = TfidfVectorizer()
tfidf_matrix = tf.fit_transform(content_df['Category_place__province'])
print(tfidf_matrix.shape)  # Output: (437, 25)
``` 

Next, cosine similarity is computed between all pairs of places using the TF-IDF matrix. This similarity measure quantifies how alike two places are based on their combined category and province attributes, with higher values indicating greater similarity.

```python
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=content_df['Place_Name'], columns=content_df['Place_Name'])
``` 

To generate recommendations, the model retrieves the top-N most similar places for a given input place by sorting the cosine similarity scores. For example, when selecting "Waterpark Kenjeran Surabaya," the model recommends five other places categorized as amusement parks in the same province, demonstrating its ability to capture relevant, localized content similarity.

```python
def place_recommendations(place_name, similarity_data=cosine_sim_df, items=content_df[['Place_Name', 'Category_place__province']], k=5):
    index = similarity_data.loc[:, place_name].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(place_name, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)
```

**Recommendation Output:**

Tempat yang telah dikunjungi
| Place\_Name                 | Category      | place\_province |
| --------------------------- | ------------- | --------------- |
| Waterpark Kenjeran Surabaya | Taman Hiburan | Jawa Timur      |

Rekomendasi Tempat Lainnya
| Place\_Name         | Category\_place\_\_province |
| ------------------- | --------------------------- |
| Surabaya North Quay | Taman Hiburan\_\_Jawa Timur |
| Taman Prestasi      | Taman Hiburan\_\_Jawa Timur |
| Taman Pelangi       | Taman Hiburan\_\_Jawa Timur |
| Ciputra Waterpark   | Taman Hiburan\_\_Jawa Timur |
| Air Mancur Menari   | Taman Hiburan\_\_Jawa Timur |

**Pros and Cons**

While content-based filtering excels in interpretability and personalization, its limitation lies in the dependency on descriptive place features. It may not effectively recommend novel or diverse destinations if user history or place attributes are sparse or highly correlated.

### Model 2 - Collaborative Filtering
The second approach in this recommendation system uses collaborative filtering, a method that predicts a user's interests by leveraging patterns in user-item interactions—specifically ratings provided by other users. This method is especially valuable when content metadata is limited or when user behavior signals are more informative than item descriptions.

Collaborative filtering was implemented using a neural network-based architecture called RecommenderNet, which learns embeddings for users and places based on their interactions.

To begin, user and place identifiers were label-encoded to convert categorical strings into numerical format:
```python
from sklearn.preprocessing import LabelEncoder

user_enc = LabelEncoder()
place_enc = LabelEncoder()

df['user'] = user_enc.fit_transform(df['User_Id'])
df['place'] = place_enc.fit_transform(df['Place_Id'])
```

The dataset was then split into training and testing sets (80/20). Then, the RecommenderNet model was defined with embedding layers to learn dense vector representations of users and places. The dot product of these embeddings was used to estimate rating scores:
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten

class RecommenderNet(Model):
    def __init__(self, n_users, n_places, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = Embedding(n_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.place_embedding = Embedding(n_places, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.dot = Dot(axes=1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[0])
        place_vector = self.place_embedding(inputs[1])
        dot_user_place = self.dot([user_vector, place_vector])
        return Flatten()(dot_user_place)
```

The model was compiled and trained using Mean Squared Error (MSE) loss and the Adam optimizer. Early stopping was applied to avoid overfitting:
```python
model = RecommenderNet(n_users, n_places)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    x=[X_train['user'], X_train['place']],
    y=y_train,
    validation_data=([X_test['user'], X_test['place']], y_test),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)
```

To generate recommendations, the model predicted scores for places the user had not yet visited. These were ranked, and the top N items were recommended:
```python
# Predict for unseen places
preds = model.predict([user_input, place_input], verbose=0)
top_idx = preds.flatten().argsort()[-10:][::-1]
top_place_indexes = candidate_places[top_idx]
recommended_place_ids = place_enc.inverse_transform(top_place_indexes)
```

**Output Example**
User Information
User ID : 34
Age : 31
Location : Sragen, Jawa Tengah

Top 5 Places Already Visited by User
| Place\_Name          | Place\_Ratings | Category      | City       |
| -------------------- | -------------- | ------------- | ---------- |
| Jembatan Kota Intan  | 5              | Budaya        | Jakarta    |
| Pulau Semak Daun     | 5              | Bahari        | Jakarta    |
| Stone Garden Citatah | 4              | Taman Hiburan | Bandung    |
| Museum Mpu Tantular  | 4              | Budaya        | Surabaya   |
| Museum Gunung Merapi | 4              | Budaya        | Yogyakarta |

Top 10 New Place Recommendations
| Place\_Name                       | Category      | City       |
| --------------------------------- | ------------- | ---------- |
| Keraton Surabaya                  | Budaya        | Surabaya   |
| Bukit Jamur                       | Cagar Alam    | Bandung    |
| Air Terjun Kedung Pedut           | Cagar Alam    | Yogyakarta |
| Desa Wisata Gamplong              | Taman Hiburan | Yogyakarta |
| Sanghyang Heuleut                 | Cagar Alam    | Bandung    |
| Monumen Yogya Kembali             | Budaya        | Yogyakarta |
| Monumen Jalesveva Jayamahe        | Budaya        | Surabaya   |
| Pantai Baron                      | Bahari        | Yogyakarta |
| Geoforest Watu Payung Turunan     | Cagar Alam    | Yogyakarta |
| Masjid Agung Trans Studio Bandung | Tempat Ibadah | Bandung    |

**Pros and Cons**

Collaborative filtering is effective when user feedback (like ratings) is abundant. It can surface unexpected recommendations beyond content similarity. However, its effectiveness is limited in cold-start scenarios (e.g., new users or items) and it can be sensitive to data sparsity.

## Evaluation

To assess the performance of the collaborative filtering model (RecommenderNet), two evaluation metrics are used: Mean Squared Error (MSE) and Mean Absolute Error (MAE). Both metrics are standard in regression tasks and are well-suited for rating prediction problems in recommendation systems.

### **Mean Squared Error (MSE)**
MSE measures the average of the squares of the errors between the predicted and actual values. It penalizes larger errors more than smaller ones due to squaring. The formula is:

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

where is $$y_i$$  the actual rating, and $$\hat{y}_i$$ is the predicted rating.

A practical interpretation of MSE is: if MSE is 2.13, then the average squared difference between predicted and actual ratings is 2.13. Although the scale is squared, it still reflects how far off predictions are on average.

### **Mean Absolute Error (MAE)** 
MAE computes the average of the absolute differences between predicted and true values. It is easier to interpret because it is in the same unit as the ratings. The formula is:

$$
MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

For example, an MAE of 1.24 indicates that, on average, the model's predictions differ from the actual ratings by 1.24 points on the 1–5 rating scale.

### **Evaluation Results**

After training the RecommenderNet model, predictions were generated on the test dataset. Since the model outputs were not naturally bounded within the 1–5 rating range, a min-max normalization step was applied to scale the predictions accordingly. This ensured comparability with the actual ratings.

The evaluation metrics on the test set are:

* Test MSE: 2.1313

* Test MAE: 1.2463

These results indicate that, on average, the predicted ratings deviate by approximately 1.25 rating points from the actual values. The MSE further confirms that the model’s errors are moderately distributed but not extremely large.

### **Model Training History**

**Image 5 Metric Visualization**
<div align="left">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/master/img/eval_history.jpg" alt="img" width="50%">
</div>
<br>

During model training, Image 5 shows a consistent reduction in both training and validation MAE. The validation MAE decreased from approximately 3.10 to 1.22 over the course of training, aligning with the test MAE result and indicating good generalization to unseen data.

## References
Permana, K. E., Rahmat, A. B., Wicaksana, D. A., & Ardianto, D. (2024). Collaborative filtering-based Madura Island tourism recommendation system using RecommenderNet. BIO Web of Conferences, 146, 01080. https://doi.org/10.1051/bioconf/202414601080

Pratama, D. E., Nurjanah, D., & Nurrahmi, H. (2023). Tourism Recommendation System using Weighted Hybrid Method in Bali Island. JURNAL MEDIA INFORMATIKA BUDIDARMA, 7(3), 1189. https://doi.org/10.30865/mib.v7i3.6409

Samara, D., Magnisalis, I., & Peristeras, V. (2020). Artificial intelligence and big data in tourism: A systematic literature review. Journal of Hospitality and Tourism Technology, 11(2), 343–367. https://doi.org/10.1108/jhtt-12-2018-0118

The Jakarta Post. (2018, October 23). Indonesian tourism set to beat Thailand in 5 years. The Jakarta Post. https://www.thejakartapost.com/news/2018/10/23/indonesian-tourism-set-to-beat-thailand-in-5-years.html

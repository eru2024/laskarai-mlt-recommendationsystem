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

The dataset used in this project is sourced from the Indonesia Tourism Destination dataset available on Kaggle ([link](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination?select=tourism_with_id.csv)). This dataset contains detailed information on 437 unique tourist attractions located across five major Indonesian cities, data for 300 users, and a ratings dataset comprising 10,000 entries of user-place interactions. This rich dataset provides an opportunity to develop a recommendation system that leverages both descriptive features of destinations and user-generated ratings.

### Variable Description
This dataset also consists of 3 variables, namely:
1. place: contains information on 437 tourist attractions across 5 major cities in Indonesia
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

The first graph in Image 1, "Count of Places by Category," details the categorical distribution of tourist destinations. The dataset’s 'Category' variable contains six distinct categories, with the most represented being 'Taman Hiburan' (135 places), followed by 'Budaya' (117), 'Cagar Alam' (106), 'Bahari' (47), 'Tempat Ibadah' (17), and 'Pusat Perbelanjaan' (15). This categorical breakdown indicates a significant concentration of entertainment, cultural, and natural attractions within the dataset, with other categories being less prominent.

The second graph in Image 1, "Count of Places by City," illustrates the geographical distribution of these attractions across five major Indonesian cities. The 'City' variable shows that Yogyakarta and Bandung contain the most tourist destinations, with 126 and 124 places respectively. Jakarta contains 84, Semarang 57, and Surabaya 46. This city-based analysis reveals that Yogyakarta and Bandung are primary locations for tourist destinations within this dataset, offering a broader range of attractions compared to Jakarta, Semarang, and Surabaya.

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

Based on Image 3, the largest age group is 18–24 years (87 users), followed by 30–34 (75 users) and 25–29 (74 users), with the smallest group being 35–40 years (64 users). This distribution aligns with the mean user age of 28.7 years.

**Image 4 Rating Distribution**
<div align="left">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/master/img/rating_rating.jpg" alt="img" width="50%">
</div>
<br>

Based on Image 4, Ratings are fairly evenly spread across 2, 3, 4, and 5, each with over 2,000 entries. Ratings of 1 are less frequent (~1,700). The mean rating is 3.07, indicating a general tendency for positive feedback.

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
   - `Location`: User locations with 28 distinct values, some possibly inconsistent of user's location name.
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

### Data Preparation for Content-Based Filtering
#### Feature Extraction by TF-IDF
The content-based filtering model recommends places similar to those a user has visited, using place details. The first step is to create a new column in the dataset called Category_place__province, which combines the place’s category and its province. This combination helps the model consider both the type of place and its location. This is done with the code:

```python
content_df = place_prep.copy()
content_df['Category_place__province'] = content_df['Category'] + '__' + content_df['place_province']
```

Next, the TF-IDF vectorizer is applied to this combined text feature. TF-IDF stands for Term Frequency-Inverse Document Frequency, a technique that turns text into numbers. It calculates how important a word (like “Taman Hiburan__Jawa Timur”) is across all places. This results in a matrix where each row represents a place and each column represents a unique word. The cell values tell how important that word is for the place. This step uses the following code:

```python
tf = TfidfVectorizer()
tfidf_matrix = tf.fit_transform(content_df['Category_place__province'])
print(tfidf_matrix.shape)  # Output: (437, 25)
```

This TF-IDF matrix (created by the code above) is essential for calculating similarity between places. In the modeling step, the model will compute cosine similarity between the rows of the TF-IDF matrix. Cosine similarity measures how similar two places are based on their combined category and province. The closer the value is to 1, the more similar the places are. This connection between the TF-IDF matrix and cosine similarity forms the basis for the recommendation model.

### Data Preparation for Collaborative Filtering

The collaborative filtering model predicts user preferences by learning from past user-place interactions, such as ratings. Before training, the raw data must be processed into a suitable numerical format.

#### Encoding
User IDs and place IDs are originally text labels, which cannot be directly used by machine learning models. These categorical strings are converted into numerical values using Label Encoding. This process assigns a unique integer to each user and each place.

```python
from sklearn.preprocessing import LabelEncoder

user_enc = LabelEncoder()
place_enc = LabelEncoder()

df['user'] = user_enc.fit_transform(df['User_Id'])
df['place'] = place_enc.fit_transform(df['Place_Id'])
```

Code snippet above shows that:
- `LabelEncoder()` from scikit-learn converts textual IDs into numeric labels.
- `fit_transform()` learns the unique labels and transforms them into integers.
- This encoding creates new columns user and place in the dataset, which contain numerical identifiers for each user and place.

#### Data Split

The dataset is then divided into two parts:
1. Training set (80%) for learning the patterns in user-place interactions.
2. Testing set (20%) to evaluate the model’s performance on unseen data.

Splitting ensures the model is validated properly and helps prevent overfitting.

```python
# Features and target
X = df[['user', 'place']]
y = df['Place_Ratings']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Code snippet above shows that:
- Train_test_split splits the encoded user and place pairs (X) and their ratings (y).
- `test_size=0.2` reserves 20% of data for testing.
- `random_state=42` fixes the random seed for reproducibility.

## Modeling
### Model 1 - Content-Based Filtering

After preparing the TF-IDF matrix that represents place features, the next step is to build the recommendation model.

The model uses cosine similarity to measure how similar each place is to every other place based on their TF-IDF representations. Cosine similarity compares the angle between two vectors, where a higher score (closer to 1) means the places share more similar characteristics (like category and province). This calculation is performed with the code:

```python
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=content_df['Place_Name'], columns=content_df['Place_Name'])
``` 

After calculating similarities between places, the place_recommendations function retrieves the most similar places to a given input.

```python
def place_recommendations(place_name, similarity_data=cosine_sim_df, items=content_df[['Place_Name', 'Category_place__province']], k=5):
    index = similarity_data.loc[:, place_name].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(place_name, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)
```

Here’s what happens step-by-step inside this function:

1. `similarity_data.loc[:, place_name]` retrieves all similarity scores for the selected place.
2. `.to_numpy()` converts these similarity scores into a NumPy array for easier handling.
3. `argpartition(range(-1, -k, -1))` finds the indices of the top-k most similar places by selecting the largest similarity scores.
4. `closest = similarity_data.columns[index[-1:-(k+2):-1]]` retrieves the names of these most similar places based on the indices.
5. `closest = closest.drop(place_name, errors='ignore')` removes the input place itself from the list of recommendations to avoid self-recommendation.
6. `merge(items)` attaches additional place information (such as category and province) to the recommendation list.
7. `head(k)` returns the top-k recommendations.

This function efficiently returns a ranked list of the k most similar places based on the content features (category and province). For instance, if a user chooses "Waterpark Kenjeran Surabaya," the function returns the top 5 similar places.

By applying cosine similarity on the TF-IDF matrix and extracting the top matches, the model can generate personalized, content-based recommendations using only place features. This approach enables the system to provide meaningful suggestions without relying on explicit user interaction data.

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

The Collaborative Filtering model predicts a user’s preferences by learning patterns from ratings provided by different users. This approach does not depend on the features or descriptions of the places, but rather focuses on how users have rated them.

The model is built using a deep learning architecture called `RecommenderNet`. Below is a breakdown of its structure and training process.

#### Model Structure – RecommenderNet
The RecommenderNet model consists of the following layers and components:
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

The model structure based on the code snippet above consists of:
- **Input Layers**: The model takes user IDs and place IDs (as integers) as input.
- **Embedding Layers**: Map user and place IDs into 50-dimensional vectors. This allows the model to learn relationships in a compact space.
    - Each user and place is represented as a dense vector of size 50 (controlled by embedding_size=50).
    - Embeddings are initialized using He normal initialization for efficient training.
    - L2 regularization (l2(1e-6)) is applied to embeddings to prevent overfitting.
- **Dot Layer**: Computes the dot product between the user and place embeddings to quantify their interaction (a predicted rating or preference score). Calculates the similarity between the user and place embeddings, producing a predicted rating.
- **Flatten Layer**: Converts the output into a scalar value representing the predicted rating by converting the dot product output into a flat structure (1D tensor). 

This model architecture is design to effectively learn complex, hidden patterns in user-place interactions.

#### Compile and Training Model

As shown in the code snippet below, the model is compiled to define the objective of learning (loss function), the optimization method, and performance metrics. The Mean Squared Error (MSE) is used as the loss function to measure the difference between predicted and actual ratings. The Adam optimizer is applied for efficient training, and Mean Absolute Error (MAE) is used as an additional performance metric.

```python
model = RecommenderNet(n_users, n_places)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
```

As shown in the code snippet below, the model is trained using the training dataset, with a `batch size` of 64 and for up to 100 `epochs`. `EarlyStopping` is implemented to prevent overfitting, stopping the training process if the validation loss does not improve for 5 consecutive epochs.

```python
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

#### Generating Recommendations
After training, the model predicts scores for user-place pairs that have not been visited. The predictions are sorted, and the top-N recommended places are selected.

```python
# Predict for unseen places
preds = model.predict([user_input, place_input], verbose=0)
top_idx = preds.flatten().argsort()[-10:][::-1]
top_place_indexes = candidate_places[top_idx]
recommended_place_ids = place_enc.inverse_transform(top_place_indexes)
```

Explanation:
- `user_input` and `place_input` represent the user and candidate places to score.
- `preds` contains the predicted scores; the highest scores indicate the most likely recommendations.
- `argsort()` sorts predictions; `inverse_transform()` decodes indices back to place names.

**Output Example**

User Information
* User ID : 34
* Age : 31
* Location : Sragen, Jawa Tengah

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


Recommendation output shows relevant destinations recommendation related to the visited destinations. For example, visited destinations show that user mostly visited Taman Hiburan and Budaya destinations and the recommendation gives unvisited destinations that come from Budaya and Taman Hiburan category. This shows that the recommendation system using Collaborative Filtering approach give good recommendation to the user preferences.

**Pros and Cons**

Collaborative filtering is effective when user feedback (like ratings) is abundant. It can surface unexpected recommendations beyond content similarity. However, its effectiveness is limited in cold-start scenarios (e.g., new users or items) and it can be sensitive to data sparsity.

## Evaluation

### Evaluation Model 1 - Content-based Filtering
For evaluating the content-based filtering model, several key metrics can be used to assess the quality of recommendations. These metrics focus on how relevant and well-ranked the recommended items are. Below is an explanation of each metric, the formula, the code used, and the interpretation.

#### Precision@K
Precision@K measures how many recommended items in the top-K are actually relevant. It is calculated by taking the number of relevant items in the recommended list (intersecting the recommended and actual visited places) and dividing it by K. The formula is: 

$$\text{Recall@K} = \frac{|\text{Recommended@K} \cap \text{Actual}|}{|\text{Actual}|}$$

To calculate this metric, the code below first takes the top-K recommendations from `recommended_place_names`, converts them into a set, and intersects it with the set of `actual_visited_places`. This value is then divided by K to give the precision score. A higher precision indicates more accurate recommendations. However, it doesn’t account for how many relevant items were missed.

```python
def precision_at_k(recommended_ids, actual_ids, k):
    recommended_at_k = list(recommended_ids)[:k]
    hits = len(set(recommended_at_k) & set(actual_ids))
    return hits / k
```

#### Recall@K
Recall@K looks at how many of the actual relevant items (places a user has actually visited) were correctly recommended in the top-K. It is calculated by dividing the number of relevant recommended items by the total number of actual relevant items. The formula is:

$$\text{Recall@K} = \frac{|\text{Recommended@K} \cap \text{Actual}|}{|Actual|}$$

To calculate this metric, The code below uses a similar approach to precision, but the denominator is the length of `actual_visited_places`. A higher recall means more of the user’s true interests were captured, but it might sacrifice precision.
```python
def recall_at_k(recommended_ids, actual_ids, k):
    recommended_at_k = list(recommended_ids)[:k]
    hits = len(set(recommended_at_k) & set(actual_ids))
    return hits / len(actual_ids) if actual_ids else 0
```
#### F1-Score@K
F1-Score@K balances precision and recall by taking their harmonic mean. The formula is:

$$\text{F1} = 2 * \frac{Precision * Recall}{Precision / Recall}$$

This metric provides a single score to show how well the model balances correctness and completeness. To calculate this metric, The code below calls `f1_at_k(precision, recall)` to compute this. A high F1-Score means the model is both accurate and comprehensive.
```python
def f1_at_k(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
```
#### Mean Average Precision (MAP)
Mean Average Precision (MAP@K) calculates the average of precision scores at each relevant item position within the top-K. It rewards models that recommend relevant items earlier in the list. The formula is:

$$\text{MAP} = \frac{1}{|\text{Actual}|} \sum_{i=1}^{K} P(i) \times \text{rel}(i)$$

Where:
* $P(i)$ is the **precision at position** $i$.
* $\text{rel}(i)$ is 1 if the item at position $i$ is **relevant**, otherwise it is 0.

To calculate this metric, The code below iterates through the recommended list, accumulates precision at relevant positions, and averages over the number of relevant items.
```python
def average_precision(recommended_ids, actual_ids, k):
    score = 0.0
    hits = 0
    for i, item in enumerate(list(recommended_ids)[:k]):
        if item in actual_ids:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual_ids), k) if actual_ids else 0
```

#### NDCG (Normalized Discounted Cumulative Gain)
NDCG evaluates the ranking quality by giving higher scores to relevant items appearing earlier in the recommendation list. It discounts relevant items by their position.

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Where:
* $$\text{DCG@K} = \sum_{i=1}^K \frac{rel(i)}{\log_2(i+1)}$$
* $$\text{IDCG@K} = \sum_{i=1}^{|\text{Actual}|} \frac{1}{\log_2(i+1)}$$
* $\text{rel}_i = 1$ if the item at position $i$ is relevant, else 0.

NDCG tells how good a ranked list of recommendations is by considering both the relevance of items and their order. It starts with DCG (Discounted Cumulative Gain), which adds up the relevance of each recommended item, but gives higher weight to items that appear near the top of the list. Then, IDCG (Ideal DCG) is the maximum possible DCG if all relevant items were at the top. It helps compare how close the list is to perfect. Finally, NDCG@K divides DCG by IDCG to get a score between 0 and 1 that means 1 means a perfect ranking (all relevant items at the top) and closer to 0 means a poor ranking.

To calculate this metric, The code below sums discounted gains for recommended relevant items and normalizes by the ideal DCG.

```python
def ndcg_at_k(recommended_ids, actual_ids, k):
    dcg = 0.0
    for i, item in enumerate(list(recommended_ids)[:k]):
        if item in actual_ids:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual_ids), k)))
    return dcg / idcg if idcg > 0 else 0
```

### Interpretation of Results
After computing these metrics using the recommendation results and actual relevant places, values closer to 1 indicate better performance. For example if user was visited Waterpark Kenjeran Surabaya, the metric values are:
- Precision@10 = 0.9 means 90% of the top-10 recommended places are relevant.
- Recall@10 = 0.0247 is low, indicating the model covers only a small portion of all relevant places.
- F1-Score@10 = 0.0480 is low due to the low recall.
- MAP@10 = 0.8664 shows most relevant items are ranked near the top.
- NDCG@10 = 0.9266 confirms good ranking quality, prioritizing relevant places early.

This indicates that the content-based filtering model provides accurate and well-ranked recommendations but misses many relevant items (low recall). This suggests good recommendation quality for a narrow set of items but could benefit from improvements to increase coverage.

### Metric Visualization

**Image 5 Metric Visualization (Content-based)**
<div align="left">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/master/img/eval_content.jpg" alt="img" width="100%">
</div>
<br>

Based on Image 5, this content-based filtering model appears to be highly precise, especially for a small number of recommendations (low K). It excels at identifying items that are very similar to what a user has already liked and placing them at the top of the recommendation list. This leads to strong Precision@K, MAP@K, and NDCG@K scores when K is small.

However, the system suffers from low recall. This means it might be limited in its ability to discover a broad range of relevant items for the user, potentially recommending only items very similar to existing preferences and missing out on novel or diverse relevant items that might be outside its narrowly defined content profile. This is a common characteristic of pure content-based systems, which can lead to a "filter bubble" effect which happens when a system shows users similar things over and over based on what users had liked before or what similar users liked. This means users might miss out on other different or new things.

### Evaluation Model 2 - Collaborative Filtering

To assess the performance of the collaborative filtering model (RecommenderNet), two evaluation metrics are used: Mean Squared Error (MSE) and Mean Absolute Error (MAE). Both metrics are standard in regression tasks and are well-suited for rating prediction problems in recommendation systems.

#### **Mean Squared Error (MSE)**
MSE measures the average of the squares of the errors between the predicted and actual values. It penalizes larger errors more than smaller ones due to squaring. The formula is:

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

where is $$y_i$$  the actual rating, and $$\hat{y}_i$$ is the predicted rating.

A practical interpretation of MSE is: if MSE is 2.13, then the average squared difference between predicted and actual ratings is 2.13. Although the scale is squared, it still reflects how far off predictions are on average.

#### **Mean Absolute Error (MAE)** 
MAE computes the average of the absolute differences between predicted and true values. It is easier to interpret because it is in the same unit as the ratings. The formula is:

$$
MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

For example, an MAE of 1.24 indicates that, on average, the model's predictions differ from the actual ratings by 1.24 points on the 1–5 rating scale.

#### **Evaluation Results**

After training the RecommenderNet model, predictions were generated on the test dataset. Since the model outputs were not naturally bounded within the 1–5 rating range, a min-max normalization step was applied to scale the predictions accordingly. This ensured comparability with the actual ratings.

The evaluation metrics on the test set are:

* Test MSE: 2.1313

* Test MAE: 1.2463

These results indicate that, on average, the predicted ratings deviate by approximately 1.25 rating points from the actual values. The MSE further confirms that the model’s errors are moderately distributed but not extremely large.

#### **Model Training History**

**Image 6 Metric Visualization (Collaborative)**
<div align="left">
    <img src="https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/master/img/eval_history.jpg" alt="img" width="50%">
</div>
<br>

During model training, Image 5 shows a consistent reduction in both training and validation MAE. The validation MAE decreased from approximately 3.10 to 1.22 over the course of training, aligning with the test MAE result and indicating good generalization to unseen data. Therefor, the recommendation system using Model 2 shows good result based on the metric (MAE) and is appropriate to give recommendation for Indonesia Tourism destinations.

## Conclusion
This project aimed to solve two key problems:
- First problem statement: How to create a personalized destination recommendation system based on destination data (like category and province).
- Second problem statement: How to recommend other destinations based on user ratings.

For the first problem, a content-based filtering model was built. It used TF-IDF to turn destination data into numbers and cosine similarity to measure how similar destinations are. This model achieved the goal of recommending destinations similar to those the user had already visited. The evaluation showed good performance in metrics like precision, recall, and NDCG. However, it was limited by the quality and availability of descriptive data for large number of destination recommendations.

For the second problem, a collaborative filtering model (RecommenderNet) was developed. This model focused on learning from user ratings without needing destination data. It achieved the goal of recommending new, potentially interesting destinations by finding patterns in user behavior. The evaluation showed good top-10 results and MAE score (1.25). This approach was useful for overcoming the limits of content-based filtering and helped users discover new places.

In summary, the content-based filtering model solved the first problem and achieved the first goal by using destination data and similarity. The collaborative filtering model solved the second problem and achieved the second goal by learning from user ratings. Both models successfully implemented the planned solutions: TF-IDF with cosine similarity for content-based filtering and deep learning for collaborative filtering.

The results showed that each approach has its strengths. Content-based filtering works well when there is good destination data, but it’s limited when it is required for more destination recommendations (more than 10 destinations). Collaborative filtering works well when there is enough user interaction data, and it can suggest new destinations. Together, these models can improve destination recommendations for Indonesian tourists and can be further improved by combining both approaches in the future.

## References
Permana, K. E., Rahmat, A. B., Wicaksana, D. A., & Ardianto, D. (2024). Collaborative filtering-based Madura Island tourism recommendation system using RecommenderNet. BIO Web of Conferences, 146, 01080. https://doi.org/10.1051/bioconf/202414601080

Pratama, D. E., Nurjanah, D., & Nurrahmi, H. (2023). Tourism Recommendation System using Weighted Hybrid Method in Bali Island. JURNAL MEDIA INFORMATIKA BUDIDARMA, 7(3), 1189. https://doi.org/10.30865/mib.v7i3.6409

Samara, D., Magnisalis, I., & Peristeras, V. (2020). Artificial intelligence and big data in tourism: A systematic literature review. Journal of Hospitality and Tourism Technology, 11(2), 343–367. https://doi.org/10.1108/jhtt-12-2018-0118

The Jakarta Post. (2018, October 23). Indonesian tourism set to beat Thailand in 5 years. The Jakarta Post. https://www.thejakartapost.com/news/2018/10/23/indonesian-tourism-set-to-beat-thailand-in-5-years.html

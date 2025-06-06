# -*- coding: utf-8 -*-
"""notebook_recommendation_system

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Fa4LYwbwhv0AW4WpVftJCVBcKyJWQ0oz

# Data Understanding

## Import Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""## Data Loading

Source: [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination?select=tourism_with_id.csv)
"""

place_df = pd.read_csv("https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/refs/heads/main/dataset/tourism_with_id.csv")
place_df.head()

user_df = pd.read_csv("https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/refs/heads/main/dataset/user.csv")
user_df.head()

rating_df = pd.read_csv("https://raw.githubusercontent.com/eru2024/laskarai-mlt-recommendationsystem/refs/heads/main/dataset/tourism_rating.csv")
rating_df.head()

print('Number of tourism places:', len(place_df))
print('Number of users:', len(user_df))
print('Number of ratings:', len(rating_df))

"""# Exploratory Data Analysis

## Variable Description

This dataset also consists of 3 variables, namely:

* `place`: contains information on tourist attractions in 5 major cities in Indonesia totaling ~400
* `user`: contains dummy user data to make recommendation features based on user
* `rating`: contains 3 columns, namely the user, the place, and the rating given, serves to create a recommendation system based on the rating

## Univariate Analysis
"""

# Custom function for summarize dataset structure
def get_dataframe_summary(dfs):
  summary_data = []

  for df_name, df in dfs.items():
    if not isinstance(df, pd.DataFrame):
      print(f"Warning: {df_name} is not a pandas DataFrame. Skipping.")
      continue

    for col_name in df.columns:
      summary_data.append({
          'DataFrame Name': df_name,
          'Column Name': col_name,
          'dtype': df[col_name].dtype,
          'Minimum Value': df[col_name].min() if pd.api.types.is_numeric_dtype(df[col_name]) else np.nan,
          'Maximum Value': df[col_name].max() if pd.api.types.is_numeric_dtype(df[col_name]) else np.nan,
          'Mean Value': df[col_name].mean() if pd.api.types.is_numeric_dtype(df[col_name]) else np.nan,
          'Median Value': df[col_name].median() if pd.api.types.is_numeric_dtype(df[col_name]) else np.nan,
          'Standard Deviation': df[col_name].std() if pd.api.types.is_numeric_dtype(df[col_name]) else np.nan,
          'Number of Rows': len(df),
          'Number of Missing Values': df[col_name].isnull().sum(),
          'Number of Unique Values': df[col_name].nunique(),
          'Number of Duplicated Values': df.duplicated(subset=[col_name]).sum()
      })

  return pd.DataFrame(summary_data)

# Apply function on avalaible dataframes
dataframes = {'place_df': place_df,
              'user_df': user_df,
              'rating_df': rating_df}

summary_dataset = get_dataframe_summary(dataframes)
print(summary_dataset)

print('Unique Categories:', place_df['Category'].unique())
print('Unique Cities:', place_df['City'].unique())

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot for Categories
sns.countplot(ax=axes[0], y='Category', data=place_df, order=place_df['Category'].value_counts().index)
axes[0].set_title('Count of Places by Category')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Category')

# Add data labels for Categories plot
for patch in axes[0].patches:
    width = patch.get_width()
    height = patch.get_height()
    x = width - width * 0.1  # position label slightly inside the bar (10% from right edge)
    y = patch.get_y() + height / 2
    axes[0].text(x, y, f'{int(width)}', va='center', ha='right', color='white', fontsize=12)

# Plot for Cities
sns.countplot(ax=axes[1], y='City', data=place_df, order=place_df['City'].value_counts().index)
axes[1].set_title('Count of Places by City')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('City')

# Add data labels for Cities plot
for patch in axes[1].patches:
    width = patch.get_width()
    height = patch.get_height()
    x = width - width * 0.1
    y = patch.get_y() + height / 2
    axes[1].text(x, y, f'{int(width)}', va='center', ha='right', color='white', fontsize=12)

plt.tight_layout()
plt.show()

# Show histogram of Price
plt.figure(figsize=(10, 6))
sns.histplot(data=place_df, x='Price', bins=30, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

"""Based on summmary dataset, **`place_df`** shows that:

* `Place_Id`: Sequential integer IDs (1–437), no missing values.
* `Place_Name` / `Description`: Text fields, each fully populated with 437 unique entries.
* `Category`: Six distinct categories.
* `City`: Five distinct cities.
* `Price`: Integer values from 0 to 900000 IDR; mean ≈ 24653, median 5000, highly right-skewed.
* `Rating`: Float values [3.4–5.0]; mean 4.44, median 4.5, low variance.
* `Time_Minutes`: Float [10–360] with 232 missing (~53%); mean 82.6, median 60, moderate variance.
* `Lat` / `Long`: Geocoordinates spanning –8.20 to +1.08 latitude and 103.93–112.82 longitude.
* `Unnamed: 11`: 100% missing—no information.
* `Unnamed: 12`: Duplicate of Place_Id.
"""

print('Unique Location:', user_df['Location'].unique())

# Define age groups
bins = [18, 25, 30, 35, 41]  # upper bounds are exclusive in cut
labels = ['18-24', '25-29', '30-34', '35-40']

# Create a new column with age groups
user_df['Age_Group'] = pd.cut(user_df['Age'], bins=bins, labels=labels, right=False)

# Get the distribution of age groups
age_group_distribution = user_df['Age_Group'].value_counts().sort_index()

# Plot the distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=age_group_distribution.index, y=age_group_distribution.values)

# Add data labels
for i, value in enumerate(age_group_distribution.values):
    plt.text(i, value + 0.5, str(value), ha='center', va='bottom')

plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.show()

"""Based on summmary dataset, **`user_df`** shows that:

* `User_Id`: Unique integer IDs (1–300), no missing values.
* `Location`: 28 distinct strings; potential typos or inconsistent naming.
* `Age`: Integer [18–40]; mean 28.7, median 29, moderate dispersion.
"""

# Show histogram of Rating
plt.figure(figsize=(10, 6))
sns.histplot(data=rating_df, x='Place_Ratings', bins=30, kde=True)
plt.title('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

"""Based on summmary dataset, **`rating_df`** shows that:

* `User_Id` / `Place_Id`: References to users and places; 10 000 rows, no missing values.
* `Place_Ratings`: Integer ratings [1–5]; mean 3.07, median 3.0.

# Data Preparation

## Missing-Value Handling

**`place_df`**
* Drop `Unnamed: 11` (100% missing) because it contains no information; retaining it adds noise and increases dimensionality without benefit.

* Drop `Unnamed: 12` (duplicate of `Place_Id`) because it is redundant identifier and preserving only one Place_Id avoids confusion in joins and modeling.

* Drop `Time_Minutes` (232/437 missing, ~53%) because it has more than half of values are missing and making row deletion is likely to bias the dataset.
"""

place_prep = place_df.drop(columns=['Unnamed: 11', 'Unnamed: 12', 'Time_Minutes']).copy()

"""## Outlier Handling

`place_df`
* Log-transform `Price` because `Price` distribution is heavily right-skewed (mean > median). Log transformation compresses high values, stabilizing variance and improving model convergence.
* Wisorize extreme `Price` becuase a few extremely expensive entries (up to 900000 IDR) can disproportionately influence distance- or similarity-based algorithms. Winsorization limits their impact while retaining relative ordering.
"""

place_prep['Price_log'] = np.log1p(place_prep['Price'])

print("Percentiles of Price:")
print(place_prep['Price'].describe(percentiles=[.25, .5, .75, .90, .95, .99]))

plt.figure(figsize=(10, 6))
sns.boxplot(x=place_prep['Price'])
plt.title('Boxplot of Price')
plt.xlabel('Price')
plt.show()

# Cap Price column at the 99th percentile
q95 = place_prep['Price'].quantile(0.95)
place_prep['Price_cap'] = place_prep['Price'].apply(lambda x: q95 if x > q95 else x)

print("\nPercentiles of Price after capping:")
print(place_prep['Price_cap'].describe(percentiles=[.25, .5, .75, .90, .95, .99]))

plt.figure(figsize=(10, 6))
sns.boxplot(x=place_prep['Price_cap'])
plt.title('Boxplot of Capped Price')
plt.xlabel('Price_cap')
plt.show()

"""**`user_df`**

* Standardize `Location` names because Inconsistent spellings (e.g., “Jakarta” vs. “DKI Jakarta”) fragment category levels, reducing encoding efficiency and potentially misleading geographic analyses.
"""

print('Unique Location:', user_df['Location'].unique())

user_prep = user_df.copy()

# Extract province from Location
user_prep['user_province'] = user_prep['Location'].str.split(', ').str[1]

# Check the result
print(user_prep[['Location', 'user_province']].head())

user_prep.info()

"""## Duplicate-Value Handling

`rating_df`

* Check and remove duplicate ratings because multiple entries for the same (User_Id, Place_Id) can skew rating distributions. Removing or aggregating duplicates ensures each interaction is counted once.
"""

# Find duplicate (User_Id, Place_Id) combinations
duplicates = rating_df[rating_df.duplicated(subset=['User_Id', 'Place_Id'], keep=False)]

# Display the duplicates
print("Duplicate User_Id and Place_Id combinations:")
print(duplicates.sort_values(by=['User_Id', 'Place_Id']))

# Group by User_Id and Place_Id, and keep the row with the highest Place_Ratings
rating_prep = rating_df.sort_values('Place_Ratings', ascending=False).drop_duplicates(
    subset=['User_Id', 'Place_Id'], keep='first'
)

# Show summary
print(f"Original rating_df shape: {rating_df.shape}")
print(f"Cleaned shape: {rating_prep.shape}")

"""## Add Columns to rating data

Add province name on `place_prep`
"""

place_prep.info()

# Mapping of City to Province
city_to_province = {
    'Jakarta': 'DKI Jakarta',
    'Yogyakarta': 'DIY',
    'Bandung': 'Jawa Barat',
    'Semarang': 'Jawa Tengah',
    'Surabaya': 'Jawa Timur'
}

# Add 'place_province' column based on 'City'
place_prep['place_province'] = place_prep['City'].map(city_to_province)

# Optional: Check for any unmapped cities
unmapped = place_prep[place_prep['place_province'].isna()]['City'].unique()
if len(unmapped) > 0:
    print("Unmapped Cities:", unmapped)

"""Add all columns from `place_prep` based on `Place_Id` in `rating_prep`"""

# Merge place_prep into rating_prep based on Place_Id
rating_prep = rating_prep.merge(place_prep, on='Place_Id', how='left')

# Check the result
rating_prep.head()

rating_prep.info()

"""Add all columns from `user_prep` based on `User_Id` in `rating_prep`"""

# Merge place_prep into rating_prep based on Place_Id
rating_prep = rating_prep.merge(user_prep , on='User_Id', how='left')

# Check the result
rating_prep.head()

rating_prep.info()

"""# Modeling

## Content Based Filtering

## Prepare Data

The data preparation step involved creating a new feature by combining the `Category` and `place_province` columns to form a contextual attribute called `Category_place__province`. This new column was intended to enrich the content-based filtering model with both thematic and geographic information for better similarity calculations.
"""

content_df = place_prep.copy()

# Create new column by combining Category and place_province
content_df['Category_place__province'] = content_df['Category'] + '__' + content_df['place_province']

# Optional: Preview the new column
print(content_df[['Category', 'place_province', 'Category_place__province']].head())

"""### TF-IDF Vectorizer

Initializing a `TfidfVectorizer` and fitting it to the `Category_place__province` column to compute the importance of each category-province combination as weighted features for content-based recommendations.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tf = TfidfVectorizer()

# Compute the IDF (Inverse Document Frequency) on the 'Category_place__province' data
tf.fit(content_df['Category_place__province'])

# Map the array from integer feature indices to feature names
tf.get_feature_names_out()

"""The TF-IDF vectorizer is applied to the `Category_place__province` column to generate a numerical matrix representation of the text data, where each row corresponds to a destination and each column represents a weighted feature.

"""

# Fit the data and then transform it into a matrix form
tfidf_matrix = tf.fit_transform(content_df['Category_place__province'])

# View the shape of the TF-IDF matrix
tfidf_matrix.shape

"""Converts the sparse TF-IDF matrix into a dense matrix format to allow easier inspection or further processing.  

"""

# Convert the TF-IDF vector into a dense matrix using the todense() function
tfidf_matrix.todense()

"""Creates a DataFrame displaying a sample of the TF-IDF matrix with selected destination names as rows and selected category-province terms as columns for easier interpretation.  

"""

# Create a DataFrame to view the TF-IDF matrix
# Columns are filled with types of cuisine
# Rows are filled with restaurant names

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=content_df['Place_Name']
).sample(22, axis=1).sample(10, axis=0)

"""### Cosine Similarity

Calculates the pairwise cosine similarity scores between all destinations based on their TF-IDF vectors.
"""

from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity on the TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

"""Creates a DataFrame representing the cosine similarity scores between tourism destinations, labeled by their names for easier interpretation and sampling.  

"""

# Create a DataFrame from the cosine_sim variable with rows and columns labeled by restaurant names
cosine_sim_df = pd.DataFrame(cosine_sim, index=content_df['Place_Name'], columns=content_df['Place_Name'])
print('Shape:', cosine_sim_df.shape)

# View the similarity matrix for a sample of restaurants
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""### Get Recommendation

Defines a function that returns the top-k most similar tourism destinations to a given place based on cosine similarity scores, excluding the place itself from the recommendations.
"""

def place_recommendations(place_name, similarity_data=cosine_sim_df, items=content_df[['Place_Name', 'Category_place__province']], k=5):
    """
    Place Recommendations based on similarity DataFrame

    Parameters:
    ---
    nama_resto : data type string (str)
        Name of the place (index of the similarity DataFrame)
    similarity_data : data type pd.DataFrame (object)
        Symmetric similarity DataFrame, with restaurants as both index and columns
    items : data type pd.DataFrame (object)
        Contains both names and other features used to define similarity
    k : data type integer (int)
        Number of recommendations to return
    ---

    For this index, we retrieve the top-k values with the highest similarity
    based on the given matrix (i).
    """

    # Retrieve data using argpartition to perform indirect partitioning along a given axis
    # Convert DataFrame to numpy array
    # Range(start, stop, step)
    index = similarity_data.loc[:, place_name].to_numpy().argpartition(
        range(-1, -k, -1))

    # Select the top-k most similar entries from the indexed result
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Drop the input restaurant name so it doesn't appear in the recommendations
    closest = closest.drop(place_name, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

"""Selects a specific place, displays its details, and prints recommended similar places based on the defined recommendation function.  

"""

pick_place = 'Waterpark Kenjeran Surabaya'
print('Tempat yang telah dikunjungi')
print(content_df[content_df['Place_Name'].eq(pick_place)][['Place_Name', 'Category', 'place_province']])
print('\nRekomendasi Tempat Lainnya')
print(place_recommendations(pick_place))

"""### Evaluate the Model"""

# Evaluation metrics functions
def precision_at_k(recommended_ids, actual_ids, k):
    recommended_at_k = list(recommended_ids)[:k]
    hits = len(set(recommended_at_k) & set(actual_ids))
    return hits / k

def recall_at_k(recommended_ids, actual_ids, k):
    recommended_at_k = list(recommended_ids)[:k]
    hits = len(set(recommended_at_k) & set(actual_ids))
    return hits / len(actual_ids) if actual_ids else 0

def f1_at_k(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

def average_precision(recommended_ids, actual_ids, k):
    score = 0.0
    hits = 0
    for i, item in enumerate(list(recommended_ids)[:k]):
        if item in actual_ids:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual_ids), k) if actual_ids else 0

def ndcg_at_k(recommended_ids, actual_ids, k):
    dcg = 0.0
    for i, item in enumerate(list(recommended_ids)[:k]):
        if item in actual_ids:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual_ids), k)))
    return dcg / idcg if idcg > 0 else 0

# Simulate a user visiting 'Waterpark Kenjeran Surabaya'
visited_place = 'Waterpark Kenjeran Surabaya'

# Get recommendations
recommended_df = place_recommendations(visited_place, k=10)
recommended_place_names = recommended_df['Place_Name'].values

# Simulate ground truth (places the user actually visited)
visited_users = rating_prep[rating_prep['Place_Name'] == visited_place]['User_Id'].unique()

# Convert to set and remove the input place to avoid including it as "relevant"
actual_visited_places = set(rating_prep[rating_prep['User_Id'].isin(visited_users)]['Place_Name'].unique())
actual_visited_places.discard(visited_place)

# Calculate Metrics
K = 10
precision = precision_at_k(recommended_place_names, actual_visited_places, K)
recall = recall_at_k(recommended_place_names, actual_visited_places, K)
f1 = f1_at_k(precision, recall)
map_score = average_precision(recommended_place_names, actual_visited_places, K)
ndcg = ndcg_at_k(recommended_place_names, actual_visited_places, K)

# Display Results
print(f"Precision@{K}: {precision:.4f}")
print(f"Recall@{K}: {recall:.4f}")
print(f"F1-Score@{K}: {f1:.4f}")
print(f"MAP@{K}: {map_score:.4f}")
print(f"NDCG@{K}: {ndcg:.4f}")

import matplotlib.pyplot as plt

# Prepare K values and metric lists
K_values = list(range(1, 21))

precision_scores = [precision_at_k(recommended_place_names, actual_visited_places, k) for k in K_values]
recall_scores = [recall_at_k(recommended_place_names, actual_visited_places, k) for k in K_values]
f1_scores = [f1_at_k(precision_scores[i], recall_scores[i]) for i in range(len(K_values))]
map_scores = [average_precision(recommended_place_names, actual_visited_places, k) for k in K_values]
ndcg_scores = [ndcg_at_k(recommended_place_names, actual_visited_places, k) for k in K_values]

# Set up subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

# Plot Precision@K
axs[0].plot(K_values, precision_scores, marker='o', color='b', label='Precision@K')
axs[0].set_title('Precision@K')
axs[0].set_xlabel('K')
axs[0].set_ylabel('Score')
axs[0].set_ylim(0, 1.05)

# Plot Recall@K
axs[1].plot(K_values, recall_scores, marker='s', color='g', label='Recall@K')
axs[1].set_title('Recall@K')
axs[1].set_xlabel('K')
axs[1].set_ylabel('Score')
axs[1].set_ylim(0, 1.05)

# Plot F1-Score@K
axs[2].plot(K_values, f1_scores, marker='^', color='r', label='F1-Score@K')
axs[2].set_title('F1-Score@K')
axs[2].set_xlabel('K')
axs[2].set_ylabel('Score')
axs[2].set_ylim(0, 1.05)

# Plot MAP@K
axs[3].plot(K_values, map_scores, marker='D', color='c', label='MAP@K')
axs[3].set_title('MAP@K')
axs[3].set_xlabel('K')
axs[3].set_ylabel('Score')
axs[3].set_ylim(0, 1.05)

# Plot NDCG@K
axs[4].plot(K_values, ndcg_scores, marker='v', color='m', label='NDCG@K')
axs[4].set_title('NDCG@K')
axs[4].set_xlabel('K')
axs[4].set_ylabel('Score')
axs[4].set_ylim(0, 1.05)

# Hide the last (unused) subplot
axs[5].axis('off')

# Layout adjustments
plt.tight_layout()
plt.show()

"""## Collaborative Filtering by Rating

### Prepare Data

Loads rating data, encodes user and place IDs into numerical labels, prepares features and target variables, then splits the data into training and testing sets.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and prepare your data
df = rating_prep.copy()
df['Place_Ratings'] = df['Place_Ratings'].astype(float)

# Encode User_Id and Place_Id
user_enc = LabelEncoder()
place_enc = LabelEncoder()

df['user'] = user_enc.fit_transform(df['User_Id'])
df['place'] = place_enc.fit_transform(df['Place_Id'])

# Define the number of unique users and places
n_users = df['user'].nunique()
n_places = df['place'].nunique()

# Features and target
X = df[['user', 'place']]
y = df['Place_Ratings']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""### Define the RecommenderNet Model

Defines a RecommenderNet model with embedding layers for users and places that computes the dot product of their embeddings to predict user-place interaction scores.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense

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
      return Flatten()(dot_user_place)  # Flatten to shape [batch_size]

"""### Compile the Model

Initializes and compiles the RecommenderNet model using mean squared error as the loss function, Adam optimizer, and mean absolute error as an evaluation metric.
"""

model = RecommenderNet(n_users, n_places)
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)

"""### Train the Model

Implements early stopping to monitor validation loss and stop training if it does not improve for 5 consecutive epochs, restoring the best model weights. Trains the model on user and place data for up to 100 epochs with a batch size of 64, using validation data for performance monitoring.
"""

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

"""### Evaluate the Model

Predicts ratings on the test set using the trained model and rescales the predictions to the original rating range of 1 to 5. Calculates the Mean Squared Error (MSE) and Mean Absolute Error (MAE) between the true ratings and the scaled predictions to evaluate model performance.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Extract test inputs and true ratings
X_test_users = X_test['user'].values
X_test_places = X_test['place'].values
y_true = y_test.values

# Predict ratings using the RecommenderNet model
y_pred = model.predict([X_test_users, X_test_places]).flatten()

# Scale y_pred to the interval (1 to 5)
min_rating = 1
max_rating = 5

y_pred_scaled = min_rating + (max_rating - min_rating) * (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())

# Calculate evaluation metrics
mse = mean_squared_error(y_true, y_pred_scaled)
mae = mean_absolute_error(y_true, y_pred_scaled)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

"""These results indicate that, on average, the predicted ratings deviate by approximately 1.25 rating points from the actual values. The MSE further confirms that the model’s errors are moderately distributed but not extremely large.

Plots the training and validation Mean Absolute Error (MAE) over epochs to visualize the model's learning progress and potential overfitting.
"""

plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training vs Validation MAE')
plt.legend()
plt.grid(True)
plt.show()

"""During model training, Image 5 shows a consistent reduction in both training and validation MAE. The validation MAE decreased from approximately 3.10 to 1.22 over the course of training, aligning with the test MAE result and indicating good generalization to unseen data.

### Make Recommendations for a User

Selects a sample user by encoded ID and retrieves their age and location information.  
Identifies places the user has visited and predicts ratings for unvisited candidate places using the trained model to generate top 10 new recommendations.  
Displays the user's top 5 highest-rated visited places alongside the recommended new places with relevant details.
"""

# Choose a sample user
user_id = 34  # must be encoded form (same as used in training)

# Get user’s real info (if available)
user_age = user_prep.loc[user_prep['User_Id'] == user_id, 'Age'].values[0]
user_location = user_prep.loc[user_prep['User_Id'] == user_id, 'Location'].values[0]

# Get places the user has already rated (visited)
visited_place_ids = rating_prep[rating_prep['User_Id'] == user_id]['Place_Id'].unique()

# Convert visited place_ids to encoded place indexes
visited_place_indexes = place_enc.transform(visited_place_ids)

# Candidate places: all places except those visited
all_place_indexes = np.arange(n_places)
candidate_places = np.setdiff1d(all_place_indexes, visited_place_indexes)

# Prepare inputs for candidate places
user_input = np.array([user_id] * len(candidate_places))
place_input = candidate_places

# Predict ratings for candidate places
preds = model.predict([user_input, place_input], verbose=0)

# Get top 10 recommendations (among candidate places)
top_idx_within_candidates = preds.flatten().argsort()[-10:][::-1]
top_place_indexes = candidate_places[top_idx_within_candidates]

# Decode encoded place indexes back to original Place_Id
recommended_place_ids = place_enc.inverse_transform(top_place_indexes)

# Get recommended place details
recommended_places = df[df['Place_Id'].isin(recommended_place_ids)][
    ['Place_Name', 'Category', 'City']
].drop_duplicates().reset_index(drop=True)

# Also show top 5 rated places by user
top_5_places = (
    rating_prep[rating_prep['User_Id'] == user_id]
    .sort_values(by='Place_Ratings', ascending=False)
    .head(5)
    [['Place_Name', 'Place_Ratings', 'Category', 'City']]
    .reset_index(drop=True)
)

# Display everything
print("USER INFORMATION")
print(f"User ID  : {user_id}")
print(f"Age      : {user_age}")
print(f"Location : {user_location}")
print("=" * 70)

print("TOP 5 PLACES ALREADY VISITED BY USER")
print(top_5_places)
print("=" * 70)

print("TOP 10 NEW PLACE RECOMMENDATIONS")
print(recommended_places)
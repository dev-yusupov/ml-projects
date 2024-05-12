import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the data
df = pd.read_csv("spotify_data.csv")
df = df.dropna(axis=0).reset_index(drop=True)


columns = ['index', 'Unnamed: 0', 'artist_name', 'track_name', 'track_id',
       'popularity', 'year', 'genre', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

scaler = StandardScaler()
numerical_cols = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
                  "liveness", "valence", "tempo", "duration_ms"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ["artist_name", "track_name", "track_id", "genre", "key"]
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


# Define features for similarity calculation
features = df[['genre', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
               'instrumentalness', 'liveness', 'valence', 'tempo']]

# Calculate similarity matrix
similarity_matrix = cosine_similarity(features, features)

# Function to decode track ID to song name and artist name
def decode_track_id(track_id):
    track_info = df[df['track_id'] == track_id]
    if not track_info.empty:
        track_name = track_info.iloc[0]['track_name']
        artist_name = track_info.iloc[0]['artist_name']
        # Fit the label encoder to the data used to generate recommendations
        label_encoder.fit(df[['track_name', 'artist_name']].values.ravel())

        # Convert track_name and artist_name to integers
        track_id_int = int(track_id)

        # Get the decoded track name and artist name
        decoded_track_name = label_encoder.inverse_transform([track_id_int])[0]
        decoded_artist_name = label_encoder.inverse_transform([track_id_int])[0]
        return decoded_track_name, decoded_artist_name
    else:
        return "Unknown", "Unknown"




# Recommendation function with similarity score threshold
def recommend_tracks(selected_tracks, n_recommendations=3, similarity_threshold=0.1):
    selected_indices = [index for index, _, _, _, _ in selected_tracks]
    avg_similarity_scores = np.mean([similarity_matrix[selected_index] for selected_index in selected_indices], axis=0)
    # Filter recommendations by similarity score threshold
    top_recommendations_indices = [idx for idx, score in enumerate(avg_similarity_scores) if score > similarity_threshold]
    top_recommendations_indices = top_recommendations_indices[-n_recommendations:][::-1]  # Select top recommendations
    recommended_tracks = [(df.iloc[index]['track_id'], avg_similarity_scores[index]) for index in top_recommendations_indices]
    return recommended_tracks



# User inputs number of random songs to select
n_random_songs = int(input("Enter the number of random songs to select: "))

# Select a random subset of the dataset for recommendation
subset_df = df.sample(n=min(n_random_songs, len(df)), replace=False)
selected_tracks = [(index, row['track_name'], row['artist_name'], row['genre'], row['year']) for index, row in subset_df.iterrows()]

# Print selected random songs
print("\nSelected random songs:")
for i, (index, track, artist, genre, year) in enumerate(selected_tracks, start=1):
    print(f"{i}. {track} by {artist} ({genre}, {year})")

# Get recommendations for selected songs
recommendations = recommend_tracks(selected_tracks)

# Print recommendations
print("\nGeneral recommendations based on the selected songs:")
for i, (track_id, similarity_score) in enumerate(recommendations, start=1):
    decoded_track_name, decoded_artist_name = decode_track_id(track_id)
    print(f"{i}. {decoded_track_name} by {decoded_artist_name} (Similarity Score: {similarity_score:.2f})")
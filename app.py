from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load the movie data with 'tags' column
new_df = pickle.load(open('new_df.pkl', 'rb'))

# Create CountVectorizer (can also use TfidfVectorizer for better accuracy)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# -------------------- Recommendation Function --------------------
def recommend(movie):
    movie = movie.lower()
    
    # 1️⃣ Check if the movie exists
    titles_lower = new_df['title'].str.lower()
    if movie not in titles_lower.values:
        # Partial match
        partial_matches = titles_lower[titles_lower.str.contains(movie, na=False)]
        if len(partial_matches) == 0:
            return []
        movie_index = partial_matches.index[0]
    else:
        movie_index = titles_lower[titles_lower == movie].index[0]

    # 2️⃣ Compute similarity only for this movie
    similarity_scores = cosine_similarity([vectors[movie_index]], vectors).flatten()

    # 3️⃣ Get top 5 recommended movies
    top_indices = similarity_scores.argsort()[-6:-1][::-1]
    return new_df.iloc[top_indices].title.tolist()

# -------------------- Routes --------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movie():
    movie = request.form['movie']
    recommendations = recommend(movie)
    if recommendations:
        return jsonify({'found': True, 'movies': recommendations})
    else:
        return jsonify({'found': False})

@app.route('/suggest', methods=['GET'])
def suggest_movies():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    suggestions = new_df['title'][new_df['title'].str.lower().str.contains(query)].tolist()[:5]
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)

import streamlit as st
from recommender import load_preprocessed_imdb_data, build_imdb_recommender, recommend_movies, recommend_by_genres
@st.cache_data
def load_data():
    movies, movie_titles = load_preprocessed_imdb_data()
    vectorizer, feature_vectors = build_imdb_recommender(movies)
    return movies, movie_titles, vectorizer, feature_vectors
movies, movie_titles, vectorizer, feature_vectors = load_data()
st.title('Система рекомендаций фильмов на основе IMDb')
st.markdown("""
    <style>
    .dataframe th, .dataframe td {
        text-align: center;
    }
    .dataframe {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
all_genres = set()
for genres in movies['genres']:
    all_genres.update(genres)
genre_list = sorted([genre for genre in all_genres if genre])
selected_genres = st.multiselect('Выберите жанры:', genre_list)
if st.button('Рекомендовать по жанрам'):
    if selected_genres:
        recommendations = recommend_by_genres(selected_genres, movies)
        if not recommendations.empty:
            st.write('Рекомендуемые фильмы:')
            recommendations_display = recommendations.copy()
            recommendations_display['genres'] = recommendations_display['genres'].apply(lambda x: ', '.join(x))
            recommendations_display['averageRating'] = recommendations_display['averageRating'].map('{:.1f}'.format)
            recommendations_display['numVotes'] = recommendations_display['numVotes'].astype(int).apply(lambda x: f"{x:,}".replace(',', ' '))
            recommendations_display['startYear'] = recommendations_display['startYear'].astype(int)
            recommendations_display['title'] = recommendations_display['primaryTitle'] + ' (' + recommendations_display['startYear'].astype(str) + ')'
            recommendations_display.reset_index(drop=True, inplace=True)
            recommendations_display.index += 1
            recommendations_display.index.name = '№'
            st.table(recommendations_display[['title', 'genres', 'averageRating', 'numVotes']])
        else:
            st.write('Нет фильмов, соответствующих выбранным жанрам.')
    else:
        st.write('Пожалуйста, выберите хотя бы один жанр.')
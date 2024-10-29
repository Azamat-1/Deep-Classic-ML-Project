import pandas as pd
import pickle

def load_and_preprocess_imdb_data():
    # Загружаем файлы данных
    basics = pd.read_csv('data/title.basics.tsv', sep='\t', na_values='\\N', dtype={'tconst': str}, encoding='utf-8')
    ratings = pd.read_csv('data/title.ratings.tsv', sep='\t', na_values='\\N', dtype={'tconst': str})
    crew = pd.read_csv('data/title.crew.tsv', sep='\t', na_values='\\N', dtype={'tconst': str})
    names = pd.read_csv('data/name.basics.tsv', sep='\t', na_values='\\N', dtype={'nconst': str})
    
    movies = basics[basics['titleType'] == 'movie']
    movies = movies.dropna(subset=['primaryTitle', 'startYear'])
    movies = movies[movies['primaryTitle'].str.len() > 1]
    movies['genres'] = movies['genres'].fillna('').apply(lambda x: x.split(',') if x != '' else [])
    
    # Объединяем данные по рейтингам
    movies = movies.merge(ratings, on='tconst', how='left')
    movies['averageRating'] = movies['averageRating'].fillna(0)
    movies['numVotes'] = movies['numVotes'].fillna(0)
    
    # Фильтрация для фильмов с низким количеством голосов
    movies = movies[(movies['averageRating'] > 0) & (movies['numVotes'] >= 1000)]
    movies = movies.merge(crew[['tconst', 'directors']], on='tconst', how='left')
    
    movies['directors'] = movies['directors'].fillna('')
    movies['director_ids'] = movies['directors'].apply(lambda x: x.split(',') if x != '' else [])
    name_dict = names.set_index('nconst')['primaryName'].to_dict()
    def get_director_names(director_ids):
        return ' '.join([name_dict.get(director_id, '') for director_id in director_ids])
    
    movies['director_names'] = movies['director_ids'].apply(get_director_names)
    movies['genres_str'] = movies['genres'].apply(lambda x: ' '.join(x))
    movies['features'] = movies['primaryTitle'] + ' ' + movies['genres_str'] + ' ' + movies['director_names']
    movies['title'] = movies['primaryTitle'] + ' (' + movies['startYear'].astype(str) + ')'
    print("Столбцы в DataFrame movies перед сохранением:", movies.columns)
    
    movies = movies.drop_duplicates(subset=['title'])
    movies.to_pickle('movies.pkl')
    movie_titles = movies['title'].tolist()
    with open('titles.pkl', 'wb') as f:
        pickle.dump(movie_titles, f)
    return movies

if __name__ == '__main__':
    load_and_preprocess_imdb_data()
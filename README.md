# Movie Recommendation System

This repository contains a **content-based movie recommendation system** built using Python. The system suggests movies based on the title and genres entered by the user. It uses TF-IDF vectorization and cosine similarity to recommend movies that are most similar to the user's input.

## Features:
- **Movie Search**: Input a movie title and receive recommendations based on title similarity.
- **Genre-based Recommendations**: Suggests movies with similar genres to the one searched.
- **Data Cleaning**: Handles special characters in movie titles and genres.
- **Efficient Recommendation Engine**: Uses scikit-learn's `TfidfVectorizer` and `cosine_similarity` for recommendations.

## Technologies Used:
- **Python**: Core programming language used.
- **Pandas**: For data loading, cleaning, and manipulation.
- **scikit-learn**: For TF-IDF vectorization and similarity calculations.
- **Numpy**: For efficient array operations.
- **Regex**: For data cleaning in movie titles.

## Dataset:
The system uses a movie dataset (`movies.csv`) containing the following key columns:
- `movieId`: Unique ID for each movie.
- `title`: The name of the movie.
- `genres`: List of genres for each movie, separated by pipes (`|`).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```

4. Ensure the `movies.csv` file is present in the same directory as the script.

## Usage

1. Run the recommendation system:
   ```bash
   python3 movie_recommendation.py
   ```

2. Enter a movie title when prompted:
   ```
   Enter a movie title to get recommendations: 
   ```

3. Select the appropriate movie from the list and receive recommendations based on similar genres.

## Example

```bash
Enter a movie title to get recommendations: Toy Story
Are you looking for (please choose a number): 
0 :  Toy Story
1 :  Toy Story 2
2 :  Toy Story 3
3 :  The LEGO Movie
4 :  A Bug's Life
Enter the number of the movie you are looking for: 0

We have the following recommendations based on genres:
         clean_title              genres_list
100      Toy Story 2             Adventure Comedy Family Fantasy
200      Finding Nemo            Adventure Animation Children Comedy
...
```

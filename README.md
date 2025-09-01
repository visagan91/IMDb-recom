Recommendation engine: IMDb data
Introduction:
This report documents the implementation of a recommendation system that suggests movies to users based of the storyline. The system works on the scrapped IMDb data of movies of the year 2024 effectively scrapping the movie’s summary and analysing the user input to recommend top movies which is near to the user preference. A Streamlit app lets users type a plot or pick a movie and get top-5 suggestions.
Data Extraction and Analysis:
The dataset to be used to work the recommendation system is scrapped from the IMDb website. All the feature movies of the year 2024 is filtered and the attributes ‘Movie Name,IMDb ID,URL,Rating,Voting Counts,Duration,Storyline’ are scrapped using selenium. The code is optimally designed to overcome IMDb website’s restrictions.The extracted data is stored in imdb_2024_list_all.csv.
Upon initial analysis the dataset if found to contains 20108 rows reflecting the same number of movies and 7 columns of attributes.
Data Preprocessing:
Data cleaning: The dataset is cleaned for any punctuation, non-alphabets. Stop words removal and lowercase normalisation is also implemented. 
Tokenisation: Uses TF-IDF used to turn the sentences into tokens. 1–2 grams (unigrams + bigrams) capture both words and short phrases
Vectorisation: TfidfVectorizer is used to produce a sparse matrix that then converts the movie storyline into numerical and is stored ‘tfidf_2024.npz’ and the resulting vectoriser is stored as ‘vectorizer.pkl’.
Recommendation engine:
Once the dataset is cleaned, the outlay for the recommendation engine is started as per the assignment ask. Cosine similarity is used to compute the similar movies the system can recommend. Since on L2-normalized TF-IDF, cosine = dot product we compute all similarities with one multiply: sims = X @ q.T.
Streamlit app:
The interface for the user is provided on a streamline with the following features
Modes:
Type a storyline (free text)
More like this movie (pick a 2024 title)
Controls in sidebar:
-How many recommendations? (1–20)
-Min similarity (tighten/relax match strictness)
-Min rating (optional metadata filter)
-Show “why it matched” terms
-Exclude the selected movie
Output:
-Ranked list with title, similarity score, optional rating
-Storyline in an expander
-Open on IMDb link when available
-Download CSV of results

Limitations:
Since TF_IDF relies on shared words and phrases, it is not optimal to catch deep patterns like synonyms and other neural language embeddings.
Page scraping structs can change and may require constant updates and maintenance. 
Improvements:
Semantic embeddings with ANN can be used to relate deep language patterns and phenomenons.
A more hybrid approach with features shared among different models for say working and space optimisation. 
Challenges faced:

Dynamic loading:
Problem : The page uses a “50 more” button and/or paginated date filters, the scraper emulates user clicks until the button disappears was devised. With kept on crashing the tab. 
Overcome: Restarted the driver very time to reduce how much is kept in the DOM (monthly slicing helps); add small sleeps between loads. And also scraping was sliced to month wise.

Conclusion:
This project delivers a practical, fast recommender by combining robust scraping with TF-IDF cosine similarity and a clean Streamlit UI. It’s easily extensible to richer text encoders, rerankers, and periodic refresh jobs for production use.
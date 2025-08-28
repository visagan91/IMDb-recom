from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# 🎬 Genres and URLs (2024 movies)
genres = {
    'Action': 'https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&genres=action',
    'Comedy': 'https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&genres=comedy',
    'Horror': 'https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&genres=horror',
    'Drama': 'https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&genres=drama',
    'Sci-Fi': 'https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&genres=sci-fi',
    'Thriller': 'https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&genres=thriller',
    'Romance': 'https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&genres=romance',
    'Animation': 'https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&genres=animation',
    'Adventure': 'https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&genres=adventure'
}

for genre_name, genre_url in genres.items():
    print(f"\n🎬 Scraping movies: {genre_name}")

    # Create fresh browser per genre
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.maximize_window()

    driver.get(genre_url)
    movies_list = []

    driver.get(genre_url)
    movies_list = []

    while True:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'li.ipc-metadata-list-summary-item'))
            )
        except:
            print("⚠️ Timeout: movies not loaded.")
            break

        movies = driver.find_elements(By.CSS_SELECTOR, 'li.ipc-metadata-list-summary-item')
        print(f"✅ Found total movies loaded so far: {len(movies)}")

        for movie in movies[len(movies_list):]:  # scrape only new ones
            try:
                name = movie.find_element(By.CSS_SELECTOR, 'h3.ipc-title__text').text
            except:
                name = ''
            try:
                rating = movie.find_element(By.CSS_SELECTOR, 'span.ipc-rating-star--rating').text
            except:
                rating = ''
            try:
                votes = movie.find_element(By.CSS_SELECTOR, 'span.ipc-rating-star--voteCount').text.strip('()').replace(',', '')
            except:
                votes = ''
            try:
                summary = movie.find_element(By.CSS_SELECTOR, '<div class="ipc-html-content-inner-div" role="presentation"></div>')    
            except:
                summary = ''    
            try:
                metadata = movie.find_elements(By.CSS_SELECTOR, 'div.dli-title-metadata span')
                duration = metadata[1].text if len(metadata) >= 2 else ''
            except:
                duration = ''

            movies_list.append({
                'Movie Name': name,
                'Genre': genre_name,
                'Rating': rating,
                'Voting Counts': votes,
                'Duration': duration,
                'Summary': summary
            })

        # Try to click "50 more" button
        try:
            more_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'svg.ipc-icon--expand-more'))
            )
            ActionChains(driver).move_to_element(more_button).click().perform()
            print("➡️ Clicked '50 more' button, loading more movies...")
            time.sleep(2)
        except:
            print("✅ No more '50 more' button found. Stopping.")
            break

    driver.quit()

    # Save CSV for this genre
    df = pd.DataFrame(movies_list)
    csv_filename = f'imdb_movies_{genre_name.lower()}_2024.csv'
    df.to_csv(csv_filename, index=False)
    print(f"📦 Saved {len(df)} movies to {csv_filename}")

print("\n✅ Finished scraping all genres!")
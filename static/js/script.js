document.addEventListener('DOMContentLoaded', function() {
    fetchDefaultMovies();
    fetchTopTenMovies();
    fetchPeopleAlsoWatchMovies();
    const searchIcon = document.getElementById('search-icon');
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const suggestions = document.getElementById('suggestions');
    const movieTitleOverlay = document.getElementById('movie-title');

    searchIcon.addEventListener('click', function() {
        searchForm.style.display = 'flex';
        searchIcon.style.display = 'none';
        searchInput.focus();
    });

    searchInput.addEventListener('blur', function() {
        if (this.value === '') {
            searchForm.style.display = 'none';
            searchIcon.style.display = 'block';
        }
    });

    searchInput.addEventListener('input', handleInput);
    document.getElementById('search-form').addEventListener('submit', handleSearch);

    // Enable dragging functionality
    makeElementDraggable(searchForm);

    // Wrap the movie title text for the details page
    if (movieTitleOverlay) {
        movieTitleOverlay.innerText = wrapText(movieTitleOverlay.innerText, 3);
    }
});

function handleInput(event) {
    const query = event.target.value;
    if (query.length > 1) {
        fetch(`/search?query=${query}`)
        .then(response => response.json())
        .then(data => {
            const suggestions = data.results.map(item => `<div class="suggestion-item" onclick="selectSuggestion('${item.title}')">${item.title}</div>`).join('');
            document.getElementById('suggestions').innerHTML = suggestions;
            document.getElementById('suggestions').style.display = 'block';

            const trailerVideo = document.getElementById('background-video');
            const movieTitleOverlay = document.getElementById('movie-title');
            if (data.trailer_url) {
                trailerVideo.src = `${data.trailer_url}?autoplay=1&loop=1&playlist=${data.trailer_url.split('embed/')[1]}`;
                movieTitleOverlay.innerText = wrapText(data.trailer_title, 3);
            }
            updateSuggestionsPosition();
        })
        .catch(error => console.error('Error fetching suggestions:', error));
    } else {
        document.getElementById('suggestions').style.display = 'none';
    }
}

function wrapText(text, maxWordsPerLine) {
    const words = text.split(' ');
    let wrappedText = '';
    for (let i = 0; i < words.length; i += maxWordsPerLine) {
        wrappedText += words.slice(i, i + maxWordsPerLine).join(' ') + '\n';
    }
    return wrappedText;
}

function selectSuggestion(title) {
    document.getElementById('search-input').value = title;
    document.getElementById('suggestions').style.display = 'none';
}

function handleSearch(event) {
    event.preventDefault();
    const query = document.querySelector('input[name="query"]').value;
    fetch(`/search?query=${query}`)
    .then(response => response.json())
    .then(data => {
        const searchResultsList = document.getElementById('search-results-list');
        searchResultsList.innerHTML = '';

        const trailerVideo = document.getElementById('background-video');
        const movieTitleOverlay = document.getElementById('movie-title');
        if (data.trailer_url) {
            trailerVideo.src = `${data.trailer_url}?autoplay=1&loop=1&playlist=${data.trailer_url.split('embed/')[1]}`;
            movieTitleOverlay.innerText = wrapText(data.trailer_title, 3);
        }

        data.results.forEach((movie, index) => {
            if (index % 6 === 0) searchResultsList.appendChild(document.createElement('div')).classList.add('movie-row');
            const movieDiv = document.createElement('div');
            movieDiv.classList.add('movie-item');
            movieDiv.innerHTML = `
                <img src="https://image.tmdb.org/t/p/w200${movie.poster_path}" alt="${movie.title}">
                <button class="view-details-button" onclick="viewMovieDetails(${movie.id})">View Details</button>
            `;
            searchResultsList.lastChild.appendChild(movieDiv);
        });
        // Fetch recommendations for the first movie in the search results
        if (data.results.length > 0) {
            const firstMovieId = data.results[0].id;
            fetch(`/recommend`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ movie_id: firstMovieId }),
            })
            .then(response => response.json())
            .then(recommendationData => {
                const recommendationsList = document.getElementById('recommendations-list');
                recommendationsList.innerHTML = '';
                recommendationData.similar_movies.forEach(movie => {
                    const movieDiv = document.createElement('div');
                    movieDiv.classList.add('movie-item');
                    movieDiv.innerHTML = `
                        <img src="https://image.tmdb.org/t/p/w200${movie.poster_path}" alt="${movie.title}">
                        <button class="watch-now-button" onclick="viewMovieDetails(${movie.id})">WATCH NOW</button>
                    `;
                    recommendationsList.appendChild(movieDiv);
                });
                document.getElementById('recommendations').style.display = 'block';
            })
    .catch(error => console.error('Error fetching search results:', error));
}
    })
    .catch(error => console.error('Error fetching search results:', error));
}

function fetchDefaultMovies() {
    fetch('/default_movies')
    .then(response => response.json())
    .then(data => {
        const trendingMoviesList = document.getElementById('trending-movies-list');
        const latestTrailersList = document.getElementById('latest-trailers-list');

        trendingMoviesList.innerHTML = '';
        latestTrailersList.innerHTML = '';

        data.trending_movies.slice(0, 100).forEach((movie, index) => {
            if (index % 100 === 0) trendingMoviesList.appendChild(document.createElement('div')).classList.add('movie-row');
            const movieDiv = document.createElement('div');
            movieDiv.classList.add('movie-item');
            movieDiv.innerHTML = `
                <img src="https://image.tmdb.org/t/p/w200${movie.poster_path}" alt="${movie.title}">
                <button class="watch-now-button" onclick="viewMovieDetails(${movie.id})">WATCH NOW</button>
            `;
            trendingMoviesList.lastChild.appendChild(movieDiv);
        });

        data.trending_movies.slice(100, 200).forEach((movie, index) => {
            if (index % 100 === 0) latestTrailersList.appendChild(document.createElement('div')).classList.add('movie-row');
            const movieDiv = document.createElement('div');
            movieDiv.classList.add('movie-item');
            movieDiv.innerHTML = `
                <img src="https://image.tmdb.org/t/p/w200${movie.poster_path}" alt="${movie.title}">
                <button class="watch-now-button" onclick="viewMovieDetails(${movie.id})">WATCH NOW</button>
            `;
            latestTrailersList.lastChild.appendChild(movieDiv);
        });
    })
    .catch(error => console.error('Error fetching default movies:', error));
}

function fetchTopTenMovies() {
    fetch('/top_ten_movies')
    .then(response => response.json())
    .then(data => {
        const topTenMoviesList = document.getElementById('top-10-movies-list');
        topTenMoviesList.innerHTML = '';

        data.top_ten_movies.forEach((movie, index) => {
            const movieDiv = document.createElement('div');
            movieDiv.classList.add('movie-item');
            movieDiv.innerHTML = `
                <div class="top-movie-number">${index + 1}</div>
                <img src="https://image.tmdb.org/t/p/w200${movie.poster_path}" alt="${movie.title}">
                <button class="recently-added-button" onclick="viewMovieDetails(${movie.id})">Recently Added</button>
            `;
            topTenMoviesList.appendChild(movieDiv);
        });
    })
    .catch(error => console.error('Error fetching top 10 movies:', error));
}

function fetchPeopleAlsoWatchMovies() {
    fetch('/people_also_watch')
    .then(response => response.json())
    .then(data => {
        const peopleAlsoWatchList = document.getElementById('people-also-watch-list');
        peopleAlsoWatchList.innerHTML = '';

        data.people_also_watch.forEach((movie, index) => {
            if (index % 50 === 0) peopleAlsoWatchList.appendChild(document.createElement('div')).classList.add('movie-row');
            const movieDiv = document.createElement('div');
            movieDiv.classList.add('movie-item');
            movieDiv.innerHTML = `
                <img src="https://image.tmdb.org/t/p/w200${movie.poster_path}" alt="${movie.title}">
                <button class="watch-now-button" onclick="viewMovieDetails(${movie.id})">WATCH NOW</button>
            `;
            peopleAlsoWatchList.lastChild.appendChild(movieDiv);
        });
    })
    .catch(error => console.error('Error fetching ' + 'People Also Watch' + ' movies:', error));
}

function viewMovieDetails(movieId) {
    window.location.href = `/movie_details/${movieId}`;
}

function makeElementDraggable(element) {
    let isDragging = false;
    let offsetX, offsetY;

    element.addEventListener('mousedown', function(e) {
        isDragging = true;
        offsetX = e.clientX - parseInt(window.getComputedStyle(element).left);
        offsetY = e.clientY - parseInt(window.getComputedStyle(element).top);
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        element.style.zIndex = 5; // Bring the search form to the top while dragging
    });

    function onMouseMove(e) {
        if (isDragging) {
            element.style.left = `${e.clientX - offsetX}px`;
            element.style.top = `${e.clientY - offsetY}px`;
            updateSuggestionsPosition();
        }
    }

    function onMouseUp() {
        isDragging = false;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        element.style.zIndex = 4; // Reset z-index after dragging
    }

    function updateSuggestionsPosition() {
        const suggestions = document.getElementById('suggestions');
        suggestions.style.left = element.style.left;
        suggestions.style.top = `${parseInt(element.style.top) + element.offsetHeight}px`;
    }
}

function checkSimilarity() {
    const word1 = document.getElementById('word1').value;
    const word2 = document.getElementById('word2').value;

    fetch('/check-similarity/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ word1: word1, word2: word2 })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('similarity-score').innerText = `Similarity: ${data.similarity}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function predictCBOW() {
    const word1 = document.getElementById('word1').value;
    const word2 = document.getElementById('word2').value;

    fetch('/predict-cbow/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ word1: word1, word2: word2 })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('middle-word').innerText = `Middle word: ${data.middle_word}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function predictSG() {
    const word1 = document.getElementById('word1').value;

    fetch('/predict-sg/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ word1: word1 })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('similar-word').innerText = `Similar word: ${data.similar_word}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}


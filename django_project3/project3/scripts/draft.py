import Levenshtein

# Function to calculate the edit distance and suggest corrections
def suggest_corrections(query, candidates):
    suggestions = []
    for candidate in candidates:
        distance = Levenshtein.distance(query.lower(), candidate.lower())
        suggestions.append((candidate, distance))
    
    suggestions.sort(key=lambda x: x[1])
    return [s[0] for s in suggestions[:5]]  
# Example candidates (words in your database or dictionary)
candidate_words = ["apple", "apples", "ape", "appreciate", "application", "appeal"]

# User's query (misspelled word)
query = "appe"

# Get the suggestions for correction
corrections = suggest_corrections(query, candidate_words)

# Display the corrections
print("Did you mean:")
for correction in corrections:
    print(correction)

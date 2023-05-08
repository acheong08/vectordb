package semantic_search_go

import (
	"github.com/acheong08/semantic-search-go/rank"
	"github.com/acheong08/semantic-search-go/typings"
	"github.com/acheong08/semantic-search-go/vectors"
)

func SemanticSearch(query []string, corpus []string, results int) ([][]typings.SearchResult, error) {
	// Encode the corpus
	var encodedCorpus typings.Tensor = make(typings.Tensor, len(corpus))
	for i, text := range corpus {
		vector, err := vectors.Encode(text)
		if err != nil {
			return [][]typings.SearchResult{}, err
		}
		// Convert vector from []float64 to [][]float64
		encodedCorpus[i] = vector
	}
	var encodedQuery typings.Tensor = make(typings.Tensor, len(corpus))
	for i, text := range query {
		vector, err := vectors.Encode(text)
		if err != nil {
			return [][]typings.SearchResult{}, err
		}
		// Convert vector from []float64 to [][]float64
		encodedQuery[i] = vector
	}
	// Semantic search
	searchResult := rank.Rank(encodedQuery, encodedCorpus, results, false)
	return searchResult, nil
}

package semantic_search_go

import (
	"github.com/acheong08/semantic-search-go/rank"
	"github.com/acheong08/semantic-search-go/typings"
	"github.com/acheong08/semantic-search-go/vectors"
)

func SemanticSearch(query string, corpus []string, results int) ([]typings.SearchResult, error) {
	// Encode the corpus
	var encodedCorpus typings.Tensor = make(typings.Tensor, len(corpus))
	for i, text := range corpus {
		vector, err := vectors.Encode(text)
		if err != nil {
			return []typings.SearchResult{}, err
		}
		// Convert vector from []float64 to [][]float64
		encodedCorpus[i] = vector
	}
	encodedQuery, err := vectors.Encode(query)
	if err != nil {
		return []typings.SearchResult{}, err
	}
	queryTensor := typings.Tensor{encodedQuery}
	// Semantic search
	searchResult := rank.Rank(queryTensor, encodedCorpus, 2)
	return searchResult[0], nil
}

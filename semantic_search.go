package vectordb

import (
	"github.com/acheong08/vectordb/rank"
	"github.com/acheong08/vectordb/typings"
	"github.com/acheong08/vectordb/vectors"
)

func SemanticSearch(query []string, corpus []string, results int, sorted bool) ([][]typings.SearchResult, error) {
	var encodedCorpus [][]float64 = make([][]float64, len(corpus))
	var err error
	for i, v := range corpus {
		encodedCorpus[i], err = vectors.Encode(v)
		if err != nil {
			return [][]typings.SearchResult{}, err
		}
	}
	var encodedQuery [][]float64 = make([][]float64, len(query))
	for i, v := range query {
		encodedQuery[i], err = vectors.Encode(v)
		if err != nil {
			return [][]typings.SearchResult{}, err
		}
	}

	// Semantic search
	searchResult := rank.Rank(encodedQuery, encodedCorpus, results, sorted)
	return searchResult, nil
}

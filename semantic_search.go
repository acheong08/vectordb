package vectordb

import (
	"github.com/acheong08/vectordb/rank"
	"github.com/acheong08/vectordb/typings"
	"github.com/acheong08/vectordb/vectors"
)

func SemanticSearch(query []string, corpus []string, results int, sorted bool) ([][]typings.SearchResult, error) {
	encodedCorpus, err := vectors.EncodeMulti(corpus)
	if err != nil {
		return [][]typings.SearchResult{}, err
	}
	encodedQuery, err := vectors.EncodeMulti(query)
	if err != nil {
		return [][]typings.SearchResult{}, err
	}
	// Semantic search
	searchResult := rank.Rank(encodedQuery, encodedCorpus, results, sorted)
	return searchResult, nil
}

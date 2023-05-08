package semantic_search_go

import (
	"github.com/acheong08/semantic-search-go/rank"
	"github.com/acheong08/semantic-search-go/typings"
	"github.com/acheong08/semantic-search-go/vectors"
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

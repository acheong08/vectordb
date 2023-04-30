package main

import (
	"github.com/acheong08/semantic-search-go/semantic_search"
	"github.com/acheong08/semantic-search-go/typings"
	"github.com/acheong08/semantic-search-go/vectors"
)

// Just a test function
func main() {
	corpus := []string{
		"Google Chrome",
		"Firefox",
		"Dumpster Fire",
		"Garbage",
		"Brave",
	}
	// Encode the corpus
	var encodedCorpus typings.Tensor = make(typings.Tensor, len(corpus))
	for i, text := range corpus {
		vector, err := vectors.Encode(text)
		if err != nil {
			panic(err)
		}
		// Convert vector from []float64 to [][]float64
		encodedCorpus[i] = vector.F64()
	}
	query := "Browser"
	encodedQuery, err := vectors.Encode(query)
	if err != nil {
		panic(err)
	}
	// Convert query from []float64 to [][]float64 (tensor)
	queryTensor := typings.Tensor{encodedQuery.F64()}
	// Semantic search
	searchResult := semantic_search.SemanticSearch(queryTensor, encodedCorpus, 2)
	// Print results
	for _, result := range searchResult[0] {
		println(corpus[result.CorpusID])
		println(result.Score)
	}
}

package main

import (
	"github.com/acheong08/vectordb/rank"
	"github.com/acheong08/vectordb/vectors"
)

func main() {
	corpus := []string{
		"Google Chrome",
		"Firefox",
		"Dumpster Fire",
		"Garbage",
		"Brave",
	}
	// Encode the corpus
	encodedCorpus, err := vectors.EncodeMulti(corpus)
	if err != nil {
		panic(err)
	}
	query := "What is a good web browser?"
	encodedQuery, err := vectors.Encode(query)
	if err != nil {
		panic(err)
	}
	// Convert query from []float64 to [][]float64 (tensor)
	queryTensor := [][]float64{encodedQuery}
	// Semantic search
	searchResult := rank.Rank(queryTensor, encodedCorpus, 2, false)
	// Print results
	for _, result := range searchResult[0] {
		println(corpus[result.CorpusID])
		println(result.Score)
	}
}

# semantic-search-go
Text similarity search for Go

## Installation
```bash
go get github.com/acheong08/semantic-search-go
```

## Example usage
Simple:
```go
package main

import (
	ssearch "github.com/acheong08/semantic-search-go"
)

func main() {
	results, _ := ssearch.SemanticSearch([]string{"I need a web browser"}, []string{"Google Chrome", "Firefox", "Dumpster Fire", "Garbage", "Brave"}, 2)
	for _, result := range results[0] {
		println(corpus[result.CorpusID])
		println(result.Score)
	}
}
```

Raw:
```go
package main

import (
	"github.com/acheong08/semantic-search-go/rank"
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
		encodedCorpus[i] = vector
	}
	query := "What is a good web browser?"
	encodedQuery, err := vectors.Encode(query)
	if err != nil {
		panic(err)
	}
	// Convert query from []float64 to [][]float64 (tensor)
	queryTensor := typings.Tensor{encodedQuery}
	// Semantic search
	searchResult := rank.Rank(queryTensor, encodedCorpus, 2)
	// Print results
	for _, result := range searchResult[0] {
		println(corpus[result.CorpusID])
		println(result.Score)
	}
}
```

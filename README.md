# semantic-search-go
Text similarity search for Go

2x faster than sentence-transformers on 500×10000

## Installation
```bash
go get github.com/acheong08/semantic-search-go
```

## Example usage
Simple:
```go
package main

import (
	semantic_search "github.com/acheong08/semantic-search-go"
)

func main() {
	corpus := []string{"Google Chrome", "Firefox", "Dumpster Fire", "Garbage", "Brave"}
	results, _ := semantic_search.SemanticSearch([]string{"I need a web browser"}, corpus, 2)
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
	searchResult := rank.Rank(queryTensor, encodedCorpus, 2, false)
	// Print results
	for _, result := range searchResult[0] {
		println(corpus[result.CorpusID])
		println(result.Score)
	}
}
```

You can store your vectors from `semantic-search-go/vectors` in a database and run `semantic-search-go/rank` as needed rather than encoding every single time.

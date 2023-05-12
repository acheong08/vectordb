package main

import (
	"fmt"

	semantic_search "github.com/acheong08/vectordb"
)

func main() {
	corpus := []string{"Google Chrome", "Firefox", "Dumpster Fire", "Garbage", "Brave"}
	results, _ := semantic_search.SemanticSearch([]string{"I need a web browser"}, corpus, 2, true)
	for _, result := range results[0] {
		println(corpus[result.CorpusID])
		fmt.Println(result.Score)
	}
}

package semantic_search_go_test

import (
	"math/rand"
	"testing"
	"time"

	search "github.com/acheong08/semantic-search-go"
	"github.com/acheong08/semantic-search-go/rank"
	"github.com/acheong08/semantic-search-go/typings"
)

func generateRandomTensor(rows, cols int) typings.Tensor {
	tensor := make(typings.Tensor, rows)

	for i := range tensor {
		tensor[i] = make([]float64, cols)
		for j := range tensor[i] {
			tensor[i][j] = rand.Float64()
		}
	}

	return tensor
}

func TestRank(t *testing.T) {
	queryEmbeddings := generateRandomTensor(50, 512)
	corpusEmbeddings := generateRandomTensor(1000, 512)
	topK := 10

	benchmarkSemanticSearch(queryEmbeddings, corpusEmbeddings, topK, t)
}

func benchmarkSemanticSearch(queryEmbeddings, corpusEmbeddings typings.Tensor, topK int, t *testing.T) {
	startTime := time.Now()
	rankResults := rank.Rank(queryEmbeddings, corpusEmbeddings, topK)
	elapsedTime := time.Since(startTime)

	t.Logf("Elapsed time for ranking %d queries against %d documents: %s", len(queryEmbeddings), len(corpusEmbeddings), elapsedTime)

	for i, results := range rankResults {
		if len(results) != topK {
			t.Errorf("Query %d: expected %d results, got %d", i, topK, len(results))
		}
		for j := 1; j < len(results); j++ {
			if results[j-1].Score < results[j].Score {
				t.Errorf("Query %d: results not sorted in descending order", i)
				break
			}
		}
	}
}

func TestSemanticSearch(t *testing.T) {
	results, err := search.SemanticSearch([]string{"I need a web browser"}, []string{"Google Chrome", "Firefox", "Dumpster Fire", "Garbage", "Brave"}, 2)
	if err != nil {
		t.Errorf("Error: %s", err)
	}
	if results[0][0].CorpusID != 0 {
		t.Errorf("Expected 0, got %d", results[0][0].CorpusID)
	}
}

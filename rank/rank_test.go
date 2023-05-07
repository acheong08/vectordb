package rank_test

import (
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/acheong08/semantic-search-go/rank"
	"github.com/acheong08/semantic-search-go/typings"
	"github.com/acheong08/semantic-search-go/vectors"
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
	queryEmbeddings := generateRandomTensor(500, 512)
	corpusEmbeddings := generateRandomTensor(10000, 512)
	topK := 10
	benchmarkSemanticSearch(queryEmbeddings, corpusEmbeddings, topK, t)
}

func benchmarkSemanticSearch(queryEmbeddings, corpusEmbeddings typings.Tensor, topK int, t *testing.T) {
	// Create a file to store the profiling data
	// f, err := os.Create("rank_cpu.prof")
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// defer f.Close()

	// Start profiling
	// err = pprof.StartCPUProfile(f)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	startTime := time.Now()
	rankResults := rank.Rank(queryEmbeddings, corpusEmbeddings, topK)
	elapsedTime := time.Since(startTime)
	// pprof.StopCPUProfile()

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

func TestResults(t *testing.T) {
	query, _ := vectors.Encode("Fruit")
	queryEmbedding := typings.Tensor{query}
	corpusEmbeddings := make(typings.Tensor, 4)
	corpusEmbeddings[0], _ = vectors.Encode("Durian")
	corpusEmbeddings[1], _ = vectors.Encode("Avocado")
	corpusEmbeddings[2], _ = vectors.Encode("Trash")
	corpusEmbeddings[3], _ = vectors.Encode("Pizza")
	topK := 2
	rankResults := rank.Rank(queryEmbedding, corpusEmbeddings, topK)
	expected_results := [][]typings.SearchResult{
		{
			{
				CorpusID: 1,
				Score:    0.5796734128286458,
			},
			{
				CorpusID: 0,
				Score:    0.28114443800798694,
			},
		},
	}
	if !reflect.DeepEqual(rankResults, expected_results) {
		t.Errorf("Expected %v, got %v", expected_results, rankResults)
	}
}

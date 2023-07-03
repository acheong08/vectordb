package rank_test

import (
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"testing"
	"time"

	"github.com/acheong08/vectordb/rank"
	"github.com/acheong08/vectordb/vectors"
)

func generateRandomTensor(rows, cols int) [][]float64 {
	tensor := make([][]float64, rows)

	for i := range tensor {
		tensor[i] = make([]float64, cols)
		for j := range tensor[i] {
			tensor[i][j] = rand.Float64()
		}
	}

	return tensor
}

func TestRank(t *testing.T) {
	queryEmbeddings := generateRandomTensor(500, 384)
	corpusEmbeddings := generateRandomTensor(10000, 384)
	topK := 10
	benchmarkSemanticSearch(queryEmbeddings, corpusEmbeddings, topK, t)
}

func benchmarkSemanticSearch(queryEmbeddings, corpusEmbeddings [][]float64, topK int, t *testing.T) {
	// Create a file to store the profiling data
	f, err := os.Create("rank_cpu.prof")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Start profiling
	err = pprof.StartCPUProfile(f)
	if err != nil {
		log.Fatal(err)
	}
	startTime := time.Now()
	rankResults := rank.Rank(queryEmbeddings, corpusEmbeddings, topK, false)
	elapsedTime := time.Since(startTime)
	pprof.StopCPUProfile()

	t.Logf("Elapsed time for ranking %d queries against %d documents: %s", len(queryEmbeddings), len(corpusEmbeddings), elapsedTime)

	for i, results := range rankResults {
		if len(results) != topK {
			t.Errorf("Query %d: expected %d results, got %d", i, topK, len(results))
		}
		// for j := 1; j < len(results); j++ {
		// 	if results[j-1].Score < results[j].Score {
		// 		t.Errorf("Query %d: results not sorted in descending order", i)
		// 		break
		// 	}
		// }
	}
}

func TestResults(t *testing.T) {
	query, _ := vectors.Encode("Fruit")
	queryEmbedding := [][]float64{query}
	corpusEmbeddings := make([][]float64, 4)
	corpusEmbeddings[0], _ = vectors.Encode("Durian")
	corpusEmbeddings[1], _ = vectors.Encode("Avocado")
	corpusEmbeddings[2], _ = vectors.Encode("Trash")
	corpusEmbeddings[3], _ = vectors.Encode("Pizza")
	topK := 2
	rankResults := rank.Rank(queryEmbedding, corpusEmbeddings, topK, true)
	if rankResults[0][0].CorpusID != 0 {
		t.Errorf("Expected 0, got %d", rankResults[0][0].CorpusID)
	}
	if rankResults[0][1].CorpusID != 1 {
		t.Errorf("Expected 1, got %d", rankResults[0][1].CorpusID)
	}
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

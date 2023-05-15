package rank

import (
	"container/heap"
	"math"
	"sync"

	"golang.org/x/exp/slices"

	"github.com/acheong08/vectordb/typings"
)

func dotProduct(a, b []float64) float64 {
	result := 0.0
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

func norm(a []float64) float64 {
	result := 0.0
	for _, v := range a {
		result += v * v
	}
	return math.Sqrt(result)
}

func cosSim(queryEmbeddings, corpusEmbeddings [][]float64, queryNorms, corpusNorms *[]float64, queryStartIdx, corpusStartIdx int, topK int, resultChan chan<- [][]typings.SearchResult) {
	numQueries := len(queryEmbeddings)
	numCorpus := len(corpusEmbeddings)
	queriesResultList := make([][]typings.SearchResult, numQueries)

	for queryItr := 0; queryItr < numQueries; queryItr++ {
		scores := make([]float64, numCorpus)
		for j := 0; j < numCorpus; j++ {
			scores[j] = dotProduct(queryEmbeddings[queryItr], corpusEmbeddings[j]) / ((*queryNorms)[queryItr] * (*corpusNorms)[j])
		}

		pq := &typings.SearchResultHeap{}
		heap.Init(pq)

		for i, score := range scores {
			if pq.Len() < topK {
				heap.Push(pq, typings.SearchResult{CorpusID: i, Score: float64(score)})
			} else if score > pq.Peek().Score {
				heap.Pop(pq)
				heap.Push(pq, typings.SearchResult{CorpusID: i, Score: float64(score)})
			}
		}

		for pq.Len() > 0 {
			result := heap.Pop(pq).(typings.SearchResult)
			result.CorpusID += corpusStartIdx
			queriesResultList[queryItr] = append(queriesResultList[queryItr], result)

		}
	}

	resultChan <- queriesResultList
}
func Rank(queryEmbeddings, corpusEmbeddings [][]float64, topK int, sorted bool) [][]typings.SearchResult {
	const queryChunkSize, corpusChunkSize = 100, 1000
	queriesResultList := make([][]typings.SearchResult, len(queryEmbeddings))
	resultChan := make(chan [][]typings.SearchResult)
	numQueries := len(queryEmbeddings)
	numCorpus := len(corpusEmbeddings)
	corpus_norms := make([]float64, numCorpus)
	for i := 0; i < numCorpus; i++ {
		corpus_norms[i] = norm(corpusEmbeddings[i])
	}
	query_norms := make([]float64, numQueries)
	for i := 0; i < numQueries; i++ {
		query_norms[i] = norm(queryEmbeddings[i])
	}
	var wg sync.WaitGroup

	for queryStartIdx := 0; queryStartIdx < len(queryEmbeddings); queryStartIdx += queryChunkSize {
		for corpusStartIdx := 0; corpusStartIdx < len(corpusEmbeddings); corpusStartIdx += corpusChunkSize {
			queryEndIdx := min(queryStartIdx+queryChunkSize, len(queryEmbeddings))
			corpusEndIdx := min(corpusStartIdx+corpusChunkSize, len(corpusEmbeddings))

			wg.Add(1)
			go func(queryStartIdx, corpusStartIdx, queryEndIdx, corpusEndIdx int) {
				defer wg.Done()
				cosSim(queryEmbeddings[queryStartIdx:queryEndIdx], corpusEmbeddings[corpusStartIdx:corpusEndIdx], &query_norms, &corpus_norms, queryStartIdx, corpusStartIdx, topK, resultChan)
			}(queryStartIdx, corpusStartIdx, queryEndIdx, corpusEndIdx)
		}
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	i := 0
	for queriesResultChunk := range resultChan {
		for j := range queriesResultChunk {
			index := j + i*queryChunkSize
			if index < len(queriesResultList) {
				queriesResultList[index] = queriesResultChunk[j]
			}
		}
		i++
	}

	if sorted {
		for idx := range queriesResultList {
			slices.SortFunc(queriesResultList[idx], func(a typings.SearchResult, b typings.SearchResult) bool {
				return a.Score > b.Score
			})
			queriesResultList[idx] = queriesResultList[idx][:topK]
		}
	}

	return queriesResultList
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

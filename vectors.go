package semantic_search_go

import (
	"container/heap"
	"math"
	"sort"
)

type Tensor [][]float64

type Callable func(Tensor, Tensor) Tensor

type SearchResult struct {
	CorpusID int
	Score    float64
}

func DotProduct(a, b []float64) float64 {
	result := 0.0
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

func Norm(a []float64) float64 {
	result := 0.0
	for _, v := range a {
		result += v * v
	}
	return math.Sqrt(result)
}

func CosSim(queryEmbeddings, corpusEmbeddings Tensor) Tensor {
	numQueries := len(queryEmbeddings)
	numCorpus := len(corpusEmbeddings)
	cosScores := make(Tensor, numQueries)

	for i := 0; i < numQueries; i++ {
		cosScores[i] = make([]float64, numCorpus)
		queryNorm := Norm(queryEmbeddings[i])
		for j := 0; j < numCorpus; j++ {
			cosScores[i][j] = DotProduct(queryEmbeddings[i], corpusEmbeddings[j]) / (queryNorm * Norm(corpusEmbeddings[j]))
		}
	}

	return cosScores
}

type SearchResultHeap []SearchResult

func (h SearchResultHeap) Len() int           { return len(h) }
func (h SearchResultHeap) Less(i, j int) bool { return h[i].Score < h[j].Score }
func (h SearchResultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *SearchResultHeap) Push(x interface{}) {
	*h = append(*h, x.(SearchResult))
}

func (h *SearchResultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

// Peek returns the smallest element from the SearchResultHeap without removing it.
func (h SearchResultHeap) Peek() SearchResult {
	return h[0]
}

func SemanticSearch(queryEmbeddings, corpusEmbeddings Tensor, queryChunkSize, corpusChunkSize, topK int) [][]SearchResult {
	queriesResultList := make([][]SearchResult, len(queryEmbeddings))

	for queryStartIdx := 0; queryStartIdx < len(queryEmbeddings); queryStartIdx += queryChunkSize {
		for corpusStartIdx := 0; corpusStartIdx < len(corpusEmbeddings); corpusStartIdx += corpusChunkSize {
			queryEndIdx := queryStartIdx + queryChunkSize
			if queryEndIdx > len(queryEmbeddings) {
				queryEndIdx = len(queryEmbeddings)
			}

			corpusEndIdx := corpusStartIdx + corpusChunkSize
			if corpusEndIdx > len(corpusEmbeddings) {
				corpusEndIdx = len(corpusEmbeddings)
			}

			cosScores := CosSim(queryEmbeddings[queryStartIdx:queryEndIdx], corpusEmbeddings[corpusStartIdx:corpusEndIdx])

			for queryItr := 0; queryItr < len(cosScores); queryItr++ {
				pq := &SearchResultHeap{}
				heap.Init(pq)

				for i := 0; i < len(cosScores[queryItr]); i++ {
					if pq.Len() < topK {
						heap.Push(pq, SearchResult{CorpusID: i, Score: cosScores[queryItr][i]})
					} else if cosScores[queryItr][i] > pq.Peek().Score {
						heap.Pop(pq)
						heap.Push(pq, SearchResult{CorpusID: i, Score: cosScores[queryItr][i]})
					}
				}

				queryID := queryStartIdx + queryItr
				for pq.Len() > 0 {
					result := heap.Pop(pq).(SearchResult)
					result.CorpusID += corpusStartIdx
					queriesResultList[queryID] = append(queriesResultList[queryID], result)
				}
			}
		}
	}

	for idx := range queriesResultList {
		sort.SliceStable(queriesResultList[idx], func(i, j int) bool {
			return queriesResultList[idx][i].Score > queriesResultList[idx][j].Score
		})
		queriesResultList[idx] = queriesResultList[idx][:topK]
	}
	return queriesResultList
}

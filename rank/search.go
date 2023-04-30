package rank

import (
	"container/heap"
	"math"
	"sort"

	"github.com/acheong08/semantic-search-go/typings"
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

func cosSim(queryEmbeddings, corpusEmbeddings typings.Tensor) typings.Tensor {
	numQueries := len(queryEmbeddings)
	numCorpus := len(corpusEmbeddings)
	cosScores := make(typings.Tensor, numQueries)

	for i := 0; i < numQueries; i++ {
		cosScores[i] = make([]float64, numCorpus)
		queryNorm := norm(queryEmbeddings[i])
		for j := 0; j < numCorpus; j++ {
			cosScores[i][j] = dotProduct(queryEmbeddings[i], corpusEmbeddings[j]) / (queryNorm * norm(corpusEmbeddings[j]))
		}
	}

	return cosScores
}

func Rank(queryEmbeddings, corpusEmbeddings typings.Tensor, topK int) [][]typings.SearchResult {
	// Defaults
	queryChunkSize := 1
	corpusChunkSize := 1

	queriesResultList := make([][]typings.SearchResult, len(queryEmbeddings))

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

			cosScores := cosSim(queryEmbeddings[queryStartIdx:queryEndIdx], corpusEmbeddings[corpusStartIdx:corpusEndIdx])

			for queryItr := 0; queryItr < len(cosScores); queryItr++ {
				pq := &typings.SearchResultHeap{}
				heap.Init(pq)

				for i := 0; i < len(cosScores[queryItr]); i++ {
					if pq.Len() < topK {
						heap.Push(pq, typings.SearchResult{CorpusID: i, Score: cosScores[queryItr][i]})
					} else if cosScores[queryItr][i] > pq.Peek().Score {
						heap.Pop(pq)
						heap.Push(pq, typings.SearchResult{CorpusID: i, Score: cosScores[queryItr][i]})
					}
				}

				queryID := queryStartIdx + queryItr
				for pq.Len() > 0 {
					result := heap.Pop(pq).(typings.SearchResult)
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

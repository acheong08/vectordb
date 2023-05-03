package rank

import (
	"container/heap"
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

func cosSim(queryEmbeddings, corpusEmbeddings typings.Tensor) typings.Tensor {
	numQueries := len(queryEmbeddings)
	numCorpus := len(corpusEmbeddings)
	cosScores := make(typings.Tensor, numQueries)

	for i := 0; i < numQueries; i++ {
		cosScores[i] = make([]float64, numCorpus)
		for j := 0; j < numCorpus; j++ {
			cosScores[i][j] = dotProduct(queryEmbeddings[i], corpusEmbeddings[j])
		}
	}

	return cosScores
}

func Rank(queryEmbeddings, corpusEmbeddings typings.Tensor, topK int) [][]typings.SearchResult {
	const queryChunkSize, corpusChunkSize = 100, 1000
	queriesResultList := make([][]typings.SearchResult, len(queryEmbeddings))

	for queryStartIdx := 0; queryStartIdx < len(queryEmbeddings); queryStartIdx += queryChunkSize {
		for corpusStartIdx := 0; corpusStartIdx < len(corpusEmbeddings); corpusStartIdx += corpusChunkSize {
			queryEndIdx := min(queryStartIdx+queryChunkSize, len(queryEmbeddings))
			corpusEndIdx := min(corpusStartIdx+corpusChunkSize, len(corpusEmbeddings))

			cosScores := cosSim(queryEmbeddings[queryStartIdx:queryEndIdx], corpusEmbeddings[corpusStartIdx:corpusEndIdx])

			for queryItr, scores := range cosScores {
				pq := &typings.SearchResultHeap{}
				heap.Init(pq)

				for i, score := range scores {
					if pq.Len() < topK {
						heap.Push(pq, typings.SearchResult{CorpusID: i, Score: score})
					} else if score > pq.Peek().Score {
						heap.Pop(pq)
						heap.Push(pq, typings.SearchResult{CorpusID: i, Score: score})
					}

					// Break the loop when topK results have been processed
					if i == topK-1 {
						break
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

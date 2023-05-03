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

func cosSim(queryEmbeddings, corpusEmbeddings typings.Tensor, queryStartIdx, corpusStartIdx int, topK int, resultChan chan<- [][]typings.SearchResult) {
	numQueries := len(queryEmbeddings)
	numCorpus := len(corpusEmbeddings)
	queriesResultList := make([][]typings.SearchResult, numQueries)

	for queryItr := 0; queryItr < numQueries; queryItr++ {
		scores := make([]float64, numCorpus)
		for j := 0; j < numCorpus; j++ {
			scores[j] = dotProduct(queryEmbeddings[queryItr], corpusEmbeddings[j])
		}

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

		for pq.Len() > 0 {
			result := heap.Pop(pq).(typings.SearchResult)
			result.CorpusID += corpusStartIdx
			queriesResultList[queryItr] = append(queriesResultList[queryItr], result)

		}
	}

	resultChan <- queriesResultList
}

func Rank(queryEmbeddings, corpusEmbeddings typings.Tensor, topK int) [][]typings.SearchResult {
	const queryChunkSize, corpusChunkSize = 100, 1000
	queriesResultList := make([][]typings.SearchResult, len(queryEmbeddings))
	resultChan := make(chan [][]typings.SearchResult)

	for queryStartIdx := 0; queryStartIdx < len(queryEmbeddings); queryStartIdx += queryChunkSize {
		for corpusStartIdx := 0; corpusStartIdx < len(corpusEmbeddings); corpusStartIdx += corpusChunkSize {
			queryEndIdx := min(queryStartIdx+queryChunkSize, len(queryEmbeddings))
			corpusEndIdx := min(corpusStartIdx+corpusChunkSize, len(corpusEmbeddings))

			go cosSim(queryEmbeddings[queryStartIdx:queryEndIdx], corpusEmbeddings[corpusStartIdx:corpusEndIdx], queryStartIdx, corpusStartIdx, topK, resultChan)
		}
	}

	for i := 0; i < len(queryEmbeddings)/queryChunkSize+1; i++ {
		queriesResultChunk := <-resultChan
		for j := range queriesResultChunk {
			index := j + i*queryChunkSize
			if index < len(queriesResultList) {
				if queriesResultList[index] == nil {
					queriesResultList[index] = queriesResultChunk[j]
				} else {
					queriesResultList[index] = append(queriesResultList[index], queriesResultChunk[j]...)
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

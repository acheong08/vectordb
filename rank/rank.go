package rank

import (
	"container/heap"
	"math"

	"golang.org/x/exp/slices"

	"github.com/acheong08/vectordb/typings"
)

func dotProduct[T typings.Float](a, b []T) T {
	result := T(0.0)
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

func norm[T typings.Float](a []T) T {
	result := T(0.0)
	for _, v := range a {
		result += v * v
	}
	return T(math.Sqrt(float64(result)))
}

func cosSim[T typings.Float](queryEmbeddings, corpusEmbeddings [][]T, queryStartIdx, corpusStartIdx int, topK int, resultChan chan<- [][]typings.SearchResult) {
	numQueries := len(queryEmbeddings)
	numCorpus := len(corpusEmbeddings)
	queriesResultList := make([][]typings.SearchResult, numQueries)

	corpus_norms := make([]T, numCorpus)
	for i := 0; i < numCorpus; i++ {
		corpus_norms[i] = norm(corpusEmbeddings[i])
	}

	for queryItr := 0; queryItr < numQueries; queryItr++ {
		scores := make([]T, numCorpus)
		query_norm := norm(queryEmbeddings[queryItr])
		for j := 0; j < numCorpus; j++ {
			scores[j] = dotProduct(queryEmbeddings[queryItr], corpusEmbeddings[j]) / (query_norm * corpus_norms[j])
		}

		pq := &typings.SearchResultHeap{}
		heap.Init(pq)

		for i, score := range scores {
			if pq.Len() < topK {
				heap.Push(pq, typings.SearchResult{CorpusID: i, Score: float32(score)})
			} else if float32(score) > pq.Peek().Score {
				heap.Pop(pq)
				heap.Push(pq, typings.SearchResult{CorpusID: i, Score: float32(score)})
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

func Rank[T typings.Float](queryEmbeddings, corpusEmbeddings [][]T, topK int, sorted bool) [][]typings.SearchResult {
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
				queriesResultList[index] = queriesResultChunk[j]
			}
		}
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

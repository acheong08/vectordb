package typings

type SearchResult struct {
	CorpusID int
	Score    float64
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

// Peek returns the smallest element from the typings.typings.SearchResultHeap without removing it.
func (h SearchResultHeap) Peek() SearchResult {
	return h[0]
}

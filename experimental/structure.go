package experimental

import "github.com/acheong08/vectordb/typings"

type Chunk struct {
	Type      string
	Documents *Documents
	Chunks    *[]Chunk
}
type Document struct {
	Text      string
	Embedding []float64
}
type Documents struct {
	Documents []Document
}

type DocumentsInterface interface {
	Add(document Document) error
	RemoveByText(text string) error
	RemoveByIndex(i int) error
	GetEmbeddings() [][]float64
	GetTexts() []string
	Get(i int) Document
	Len() int
	Query(queries []string, topK int, sorted bool) ([][]typings.SearchResult, error)
}

type ChunkInterface interface {
	SetChunk(chunk Chunk) error
	SetDocuments(documents Documents) error
	// Interface returned should be of type Documents or Chunk
	Query(query string, topK int, sorted bool) (*interface{}, error)
}
type Client struct {
	Chunk ChunkInterface
}

type ClientInterface interface {
	// This should handle encoding and finding the correct chunk to add the document to
	Add(text string) error
	// Queries return a pointer to a Documents struct. This means that you can remove documents from the returned struct and it will be reflected in the database.
	Query(queries []string, topK int, sorted bool) (*Documents, error)
	// Returns a Chunk in an optimized format. It might be difficult to write to the returned database
	GetOptimized() (Chunk, error)
}

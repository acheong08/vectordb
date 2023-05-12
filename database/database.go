package database

type Database interface {
	New(filePath string) (*Database, error)
	Store(embeddings [][]float64) error
	Load() ([][]float64, error)
	Add(embedding []float64) error
}

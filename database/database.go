package database

type Database interface {
	Store(filePath string, embeddings [][]float64) error
	Load(filePath string) ([][]float64, error)
}

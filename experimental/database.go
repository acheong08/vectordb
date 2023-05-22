package experimental

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"log"
	"os"
)

const (
	ErrorFileNotFound = "file not found"
)

type DatabaseInterface interface {
	Add(doc Unit)
	Save() error
	GetEmbeddingByIndex(i int) []float64
	GetEmbeddings() [][]float64
	GetChunks() []Chunk
}

type Database struct {
	filePath string
	Units    []Unit
}
type Unit interface {
	GetEmbeddings() []float64
	IsUnit()
	GetType() string
}

// Database functions
func (db *Database) Add(unit Unit) {
	db.Units = append(db.Units, unit)
}

func (db *Database) RemoveByIndex(i int) {
	db.Units = append(db.Units[:i], db.Units[i+1:]...)
}

func (db *Database) Save() error {
	return Store(db.filePath, db.Units)
}

func (db *Database) GetUnitByIndex(i int) Unit {
	return db.Units[i]
}

func (db *Database) GetEmbeddingByIndex(i int) []float64 {
	return db.Units[i].GetEmbeddings()
}

func (db *Database) GetEmbeddings() [][]float64 {
	embeddings := make([][]float64, len(db.Units))
	for i, doc := range db.Units {
		embeddings[i] = doc.GetEmbeddings()
	}
	return embeddings
}
func (db *Database) GetChunks() []Chunk {
	chunks := make([]Chunk, 0)
	for _, unit := range db.Units {
		if unit.GetType() == "chunk" {
			chunk, ok := unit.(*Chunk) // expect a *Chunk, not a Chunk
			if ok {
				chunks = append(chunks, *chunk)
			}
		}
	}
	return chunks
}

func NewDatabase(filePath string) (*Database, error) {
	documents, err := Load(filePath)
	if err.Error() == ErrorFileNotFound {
		return &Database{
			filePath: filePath,
			Units:    []Unit{},
		}, nil
	} else if err != nil {
		return nil, err
	}
	return &Database{
		filePath: filePath,
		Units:    documents,
	}, nil
}

func Load(filePath string) ([]Unit, error) {
	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return nil, fmt.Errorf(ErrorFileNotFound)
	}
	// Read from a file
	data, err := os.ReadFile(filePath)
	if err != nil {
		log.Println("error reading from file:", err)
		return nil, err
	}

	buf := *bytes.NewBuffer(data)
	dec := gob.NewDecoder(&buf)

	var doc []Unit
	err = dec.Decode(&doc)
	if err != nil {
		log.Println("decode error:", err)
		return nil, err
	}

	return doc, nil
}

func Store(filePath string, units []Unit) error {

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	err := enc.Encode(units)
	if err != nil {
		log.Println("encode error:", err)
		return err
	}

	// Save to a file
	err = os.WriteFile(filePath, buf.Bytes(), 0644)
	if err != nil {
		log.Println("error writing to file:", err)
		return err
	}

	return nil
}

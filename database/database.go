package database

import (
	"bytes"
	"encoding/gob"
	"log"
	"os"
)

type Database interface {
	Store(filePath string, embeddings [][]float64) error
	Load(filePath string) ([][]float64, error)
}

func Store(filePath string, embeddings [][]float64) error {

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	err := enc.Encode(embeddings)
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

func Load(filePath string) ([][]float64, error) {
	// Read from a file
	data, err := os.ReadFile(filePath)
	if err != nil {
		log.Println("error reading from file:", err)
		return nil, err
	}

	buf := *bytes.NewBuffer(data)
	dec := gob.NewDecoder(&buf)

	var doc [][]float64
	err = dec.Decode(&doc)
	if err != nil {
		log.Println("decode error:", err)
		return nil, err
	}

	return doc, nil
}

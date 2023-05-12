package gob

import (
	"bytes"
	"encoding/gob"
	"log"
	"os"
)

type Gob struct {
	filePath string
}

func New(filePath string) *Gob {
	return &Gob{
		filePath: filePath,
	}
}

func (g *Gob) Store(embeddings [][]float64) error {

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	err := enc.Encode(embeddings)
	if err != nil {
		log.Println("encode error:", err)
		return err
	}

	// Save to a file
	err = os.WriteFile(g.filePath, buf.Bytes(), 0644)
	if err != nil {
		log.Println("error writing to file:", err)
		return err
	}

	return nil
}

func (g *Gob) Load() ([][]float64, error) {
	// Read from a file
	data, err := os.ReadFile(g.filePath)
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

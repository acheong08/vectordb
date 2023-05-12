package gob_test

import (
	"math/rand"
	"os"
	"reflect"
	"testing"

	"github.com/acheong08/vectordb/database/gob"
)

func TestStoreAndLoad(t *testing.T) {

	// Create a temporary file for testing
	tmpFile, err := os.CreateTemp("", "test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name()) // clean up after test
	database := gob.New(tmpFile.Name())

	// Generate some random embeddings for testing
	embeddings := make([][]float64, 5)
	for i := range embeddings {
		embeddings[i] = make([]float64, 5)
		for j := range embeddings[i] {
			embeddings[i][j] = rand.Float64()
		}
	}

	// Test the Store function
	err = database.Store(embeddings)
	if err != nil {
		t.Fatalf("Store function returned error: %v", err)
	}

	// Test the Load function
	loadedEmbeddings, err := database.Load()
	if err != nil {
		t.Fatalf("Load function returned error: %v", err)
	}

	// Check if the loaded embeddings are the same as the original
	if !reflect.DeepEqual(embeddings, loadedEmbeddings) {
		t.Errorf("Loaded embeddings are not the same as original, got: %v, want: %v.", loadedEmbeddings, embeddings)
	}
}

package gob_test

import (
	"os"
	"testing"

	gob "github.com/acheong08/vectordb/database"
)

func TestNewDatabase(t *testing.T) {
	db, err := gob.New("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new database: %v", err)
	}
	defer os.Remove("test_db.gob")

	if len(db.Documents) != 0 {
		t.Errorf("gob.New database should not contain any documents. Got %v", db.Documents)
	}
}

func TestAdd(t *testing.T) {
	db, err := gob.New("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new database: %v", err)
	}

	doc := gob.Document{
		Text:      "Hello World",
		Embedding: []float64{0.1, 0.2, 0.3},
	}
	db.Add(doc)

	if len(db.Documents) != 1 {
		t.Errorf("Error adding document to database. Got %v", db.Documents)
	}
	if db.GetTextByIndex(0) != "Hello World" {
		t.Errorf("Error retrieving text from database. Got %s, expected Hello World", db.GetTextByIndex(0))
	}
	if db.GetEmbeddingByIndex(0)[0] != 0.1 {
		t.Errorf("Error retrieving embedding from database. Got %v, expected [0.1, 0.2, 0.3]", db.GetEmbeddingByIndex(0))
	}
}

func TestLoad(t *testing.T) {
	db, err := gob.New("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new database: %v", err)
	}
	defer os.Remove("test_db.gob")

	doc1 := gob.Document{
		Text:      "Hello",
		Embedding: []float64{0.1, 0.2, 0.3},
	}
	doc2 := gob.Document{
		Text:      "World",
		Embedding: []float64{0.4, 0.5, 0.6},
	}

	db.Add(doc1)
	db.Add(doc2)
	err = db.Save()
	if err != nil {
		t.Errorf("Error saving database: %v", err)
	}

	db2, err := gob.Load("test_db.gob")
	if err != nil {
		t.Errorf("Error loading database: %v", err)
	}

	if len(db2) != 2 {
		t.Errorf("Error loading database. Got %v", db2)
	}
	if db2[0].Text != "Hello" {
		t.Errorf("Error retrieving text from loaded database. Got %s, expected Hello", db2[0].Text)
	}
	if db2[1].Embedding[2] != 0.6 {
		t.Errorf("Error retrieving embedding from loaded database. Got %v, expected 0.6", db2[1].Embedding[2])
	}
}

func TestGetTexts(t *testing.T) {
	db, err := gob.New("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new database: %v", err)
	}

	doc1 := gob.Document{
		Text:      "Hello",
		Embedding: []float64{0.1, 0.2, 0.3},
	}
	doc2 := gob.Document{
		Text:      "World",
		Embedding: []float64{0.4, 0.5, 0.6},
	}

	db.Add(doc1)
	db.Add(doc2)

	texts := db.GetTexts()

	if len(texts) != 2 {
		t.Errorf("Error retrieving texts from database. Got %v", texts)
	}
	if texts[0] != "Hello" {
		t.Errorf("Error retrieving text from database. Got %s, expected Hello", texts[0])
	}
	if texts[1] != "World" {
		t.Errorf("Error retrieving text from database. Got %s, expected World", texts[1])
	}
}

func TestGetEmbeddings(t *testing.T) {
	db, err := gob.New("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new database: %v", err)
	}

	doc1 := gob.Document{
		Text:      "Hello",
		Embedding: []float64{0.1, 0.2, 0.3},
	}
	doc2 := gob.Document{
		Text:      "World",
		Embedding: []float64{0.4, 0.5, 0.6},
	}

	db.Add(doc1)
	db.Add(doc2)

	embeddings := db.GetEmbeddings()

	if len(embeddings) != 2 {
		t.Errorf("Error retrieving embeddings from database. Got %v", embeddings)
	}
	if len(embeddings[0]) != 3 {
		t.Errorf("Error retrieving embedding from database. Got %v, expected length 3", embeddings[0])
	}
	if embeddings[1][2] != 0.6 {
		t.Errorf("Error retrieving embedding from database. Got %v, expected 0.6", embeddings[1][2])
	}
}

func TestClose(t *testing.T) {
	db, err := gob.New("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new database: %v", err)
	}
	defer os.Remove("test_db.gob")

	doc := gob.Document{
		Text:      "Hello World",
		Embedding: []float64{0.1, 0.2, 0.3},
	}
	db.Add(doc)

	err = db.Close()
	if err != nil {
		t.Errorf("Error closing database: %v", err)
	}
}

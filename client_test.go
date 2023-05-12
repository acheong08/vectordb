package vectordb_test

import (
	"os"
	"testing"

	"github.com/acheong08/vectordb"
)

func TestAdd(t *testing.T) {
	client, err := vectordb.NewClient("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new client: %v", err)
	}

	err = client.Add("Hello World")
	if err != nil {
		t.Errorf("Error adding document to client: %v", err)
	}
}

func TestQuery(t *testing.T) {
	client, err := vectordb.NewClient("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new client: %v", err)
	}

	err = client.Add("Hello")
	if err != nil {
		t.Errorf("Error adding document to client: %v", err)
	}

	queries := []string{"Hello"}
	topK := 1
	sorted := false

	results, err := client.Query(queries, topK, sorted)
	if err != nil {
		t.Errorf("Error querying client: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Unexpected number of query results. Got %d, expected 1", len(results))
	}

	if len(results[0]) != topK {
		t.Errorf("Unexpected number of search results. Got %d, expected %d", len(results[0]), topK)
	}
}

func TestRemoveByText(t *testing.T) {
	client, err := vectordb.NewClient("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new client: %v", err)
	}

	err = client.Add("Hello")
	if err != nil {
		t.Errorf("Error adding document to client: %v", err)
	}

	err = client.RemoveByText("Hello")
	if err != nil {
		t.Errorf("Error removing document by text: %v", err)
	}
}

func TestRemoveByIndex(t *testing.T) {
	client, err := vectordb.NewClient("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new client: %v", err)
	}

	err = client.Add("Hello")
	if err != nil {
		t.Errorf("Error adding document to client: %v", err)
	}

	err = client.RemoveByIndex(0)
	if err != nil {
		t.Errorf("Error removing document by index: %v", err)
	}
}

func TestSave(t *testing.T) {
	client, err := vectordb.NewClient("test_db.gob")
	if err != nil {
		t.Errorf("Error creating new client: %v", err)
	}
	defer os.Remove("test_db.gob")

	err = client.Add("Hello")
	if err != nil {
		t.Errorf("Error adding document to client: %v", err)
	}

	err = client.Save()
	if err != nil {
		t.Errorf("Error saving client database: %v", err)
	}
}

package vectordb_test

import (
	"testing"

	search "github.com/acheong08/vectordb"
)

func TestSemanticSearch(t *testing.T) {
	results, err := search.SemanticSearch([]string{"I need a web browser"}, []string{"Google Chrome", "Firefox", "Dumpster Fire", "Garbage", "Brave"}, 2, true)
	if err != nil {
		t.Errorf("Error: %s", err)
	}
	if results[0][0].CorpusID != 0 {
		t.Errorf("Expected 1, got %d", results[0][0].CorpusID)
	}
}

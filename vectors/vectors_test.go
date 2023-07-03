package vectors_test

import (
	"testing"

	"reflect"

	"github.com/acheong08/vectordb/vectors"
)

func TestEncoding(t *testing.T) {
	result, _ := vectors.Encode("1")
	if len(result) != 768 {
		t.Errorf("Expected 768, got %v", len(result))
	}

}

func TestEncodeMulti(t *testing.T) {
	result, _ := vectors.EncodeMulti([]string{"1"})
	expected, _ := vectors.Encode("1")
	if reflect.DeepEqual(result[0][0], expected) {
		t.Errorf("Expected %v,\n\n\n\n got %v", expected, result)
	}
}

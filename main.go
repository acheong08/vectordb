package main

import (
	"fmt"

	"github.com/acheong08/semantic-search-go/vectors"
)

// Just a test function
func main() {

	text := "This is a test."

	actual, err := vectors.Encode(text)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%v", actual)
}

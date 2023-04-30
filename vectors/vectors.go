package vectors

import (
	"fmt"
	"log"
	"os"

	onnx "github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
)

func Example_gorgonia() {
	// Create a backend receiver
	backend := gorgonnx.NewGraph()
	// Create a model and set the execution backend
	model := onnx.NewModel(backend)

	// read the onnx model
	b, _ := os.ReadFile("model.onnx")
	// Decode it into the model
	err := model.UnmarshalBinary(b)
	if err != nil {
		log.Fatal(err)
	}
	// Convert string to tensor
	input_string := "Hello world"
	// Create a tensor from the string (input []byte)
	input, err := onnx.NewTensor([]byte(input_string))
	if err != nil {
		log.Fatal(err)
	}
	// Set the first input, the number depends of the model
	model.SetInput(0, input)
	err = backend.Run()
	if err != nil {
		log.Fatal(err)
	}
	// Check error
	output, _ := model.GetOutputTensors()
	// write the first output to stdout
	fmt.Println(output[0])
}

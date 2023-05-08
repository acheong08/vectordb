package vectors

import (
	"context"
	"os"

	"github.com/acheong08/semantic-search-go/typings"
	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

var home_dir string

func init() {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	var err error
	home_dir, err = os.UserHomeDir()
	if err != nil {
		home_dir = "."
	}
	// Create ~/.models directory if it doesn't exist
	if _, err := os.Stat(home_dir + "/.models"); os.IsNotExist(err) {
		os.Mkdir(home_dir+"/.models", 0755)
	}
}

func Encode(text string) ([]float64, error) {
	modelsDir := home_dir + "/.models"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"

	m, err := tasks.Load[textencoding.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	defer tasks.Finalize(m)

	result, err := m.Encode(context.Background(), text, int(bert.MeanPooling))
	if err != nil {
		return []float64{}, err
	}
	return result.Vector.Data().F64(), nil
}

func EncodeMulti(texts []string) (typings.Tensor, error) {
	modelsDir := home_dir + "/.models"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"

	m, err := tasks.Load[textencoding.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	defer tasks.Finalize(m)

	// Using Go routines to encode multiple texts in parallel

	// Create a channel to receive the results
	results := make(chan []float64, len(texts))

	// Create a channel to receive errors
	errs := make(chan error, len(texts))

	for _, text := range texts {
		go func(text string) {
			result, err := m.Encode(context.Background(), text, int(bert.MeanPooling))
			if err != nil {
				errs <- err
			}
			results <- result.Vector.Data().F64()
		}(text)
	}

	// Collect the results
	var vectors typings.Tensor = make(typings.Tensor, len(texts))
	for i := 0; i < len(texts); i++ {
		select {
		case err := <-errs:
			return typings.Tensor{}, err
		case result := <-results:
			vectors[i] = result
		}
	}
	return vectors, nil
}

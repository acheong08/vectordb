package vectors

import (
	"context"
	"os"
	"sync"

	"github.com/acheong08/cybertron/pkg/models/bert"
	"github.com/acheong08/cybertron/pkg/tasks"
	"github.com/acheong08/cybertron/pkg/tasks/textencoding"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

var home_dir string
var defaultModelName string

func init() {
	defaultModelName = os.Getenv("VECTOR_ENCODER_MODEL")
	if defaultModelName == "" {
		defaultModelName = "sentence-transformers/distiluse-base-multilingual-cased-v2"
	}
	zerolog.SetGlobalLevel(zerolog.FatalLevel)
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

	m, err := tasks.Load[textencoding.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: defaultModelName})
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

func EncodeMulti(texts []string) ([][]float64, error) {
	modelsDir := home_dir + "/.models"

	m, err := tasks.Load[textencoding.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: defaultModelName})
	if err != nil {
		return [][]float64{}, err
	}
	defer tasks.Finalize(m)

	var resultMutex sync.Mutex
	var wg sync.WaitGroup
	results := make([][]float64, len(texts))

	for i, text := range texts {
		wg.Add(1)
		go func(i int, text string) {
			defer wg.Done()
			result, err := m.Encode(context.Background(), text, int(bert.MeanPooling))
			if err != nil {
				log.Fatal().Err(err).Send()
				return
			}
			resultMutex.Lock()
			defer resultMutex.Unlock()
			results[i] = result.Vector.Data().F64()
		}(i, text)
	}

	wg.Wait()
	return results, nil
}

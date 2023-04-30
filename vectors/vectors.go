package vectors

import (
	"context"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func Encode(text string) ([]float64, error) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)

	modelsDir := "models"
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

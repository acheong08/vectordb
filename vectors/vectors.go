package vectors

import (
	"context"
	"os"
	"path"
	"path/filepath"
	"strings"

	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/nn"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

var home_dir string

func init() {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
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
	modelPath := modelsDir + "/" + modelName

	m, err := LoadTextEncoding(modelPath)
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

// TextEncoding is a text encoding model.
type TextEncoding struct {
	// Model is the model used to answer questions.
	Model *bert.ModelForSequenceEncoding
	// Tokenizer is the tokenizer used to tokenize questions and passages.
	Tokenizer *wordpiecetokenizer.WordPieceTokenizer
	// doLowerCase is a flag indicating if the model should lowercase the input before tokenization.
	doLowerCase bool
	// embeddingsRepo is the repository used for loading embeddings.
	embeddingsRepo *diskstore.Repository
}

// LoadTextEncoding returns a TextEncoding loading the model, the embeddings and the tokenizer from a directory.
func LoadTextEncoding(modelPath string) (*TextEncoding, error) {
	vocab, err := vocabulary.NewFromFile(filepath.Join(modelPath, "vocab.txt"))
	if err != nil {
		return nil, fmt.Errorf("failed to load vocabulary for text encoding: %w", err)
	}
	tokenizer := wordpiecetokenizer.New(vocab)

	tokenizerConfig, err := bert.ConfigFromFile[bert.TokenizerConfig](path.Join(modelPath, "tokenizer_config.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer config for text encoding: %w", err)
	}

	embeddingsRepo, err := diskstore.NewRepository(filepath.Join(modelPath, "repo"), diskstore.ReadOnlyMode)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings repository for text encoding: %w", err)
	}

	m, err := nn.LoadFromFile[*bert.ModelForSequenceEncoding](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load bert model: %w", err)
	}

	err = m.Bert.SetEmbeddings(embeddingsRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to set embeddings: %w", err)
	}

	return &TextEncoding{
		Model:          m,
		Tokenizer:      tokenizer,
		doLowerCase:    tokenizerConfig.DoLowerCase,
		embeddingsRepo: embeddingsRepo,
	}, nil
}

// Encode returns the dense encoded representation of the given text.
func (m *TextEncoding) Encode(_ context.Context, text string, poolingStrategy int) (textencoding.Response, error) {
	encoded, err := m.Model.Encode(m.tokenize(text), bert.PoolingStrategyType(poolingStrategy))
	defer func() {
		go ag.ReleaseGraph(encoded)
	}()
	if err != nil {
		return textencoding.Response{}, err
	}

	response := textencoding.Response{
		Vector: encoded.Value(),
	}
	return response, nil
}

// tokenize returns the tokens of the given text (including padding tokens).
func (m *TextEncoding) tokenize(text string) []string {
	if m.doLowerCase {
		text = strings.ToLower(text)
	}
	cls := wordpiecetokenizer.DefaultClassToken
	sep := wordpiecetokenizer.DefaultSequenceSeparator
	return append([]string{cls}, append(tokenizers.GetStrings(m.Tokenizer.Tokenize(text)), sep)...)
}

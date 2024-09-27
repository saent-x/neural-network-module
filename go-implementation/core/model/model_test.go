package model

import (
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/saent-x/ids-nn/core/optimization"
	"testing"
)

func TestForRegressionModel(t *testing.T) {
	X, y := core.SineData(1000)
	regression_model := New()

	regression_model.Add(layer.MockRegressionLayer(1, 64, 0, 0, 0, 0))
	regression_model.Add(new(activation.ReLU))

	regression_model.Add(layer.MockRegressionLayer(64, 64, 0, 0, 0, 0))
	regression_model.Add(new(activation.ReLU))

	regression_model.Add(layer.MockRegressionLayer(64, 1, 0, 0, 0, 0))
	regression_model.Add(new(activation.Linear))

	regression_model.Set(new(loss.MeanSquaredError), optimization.CreateAdaptiveMomentum(0.001, .0, 0.0000001, 0.9, 0.999), new(accuracy.RegressionAccuracy))

	regression_model.Finalize()

	regression_model.Train(X, y, 10000, 100)
}

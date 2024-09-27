package model

import (
	"fmt"
	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/saent-x/ids-nn/core/optimization"
	"gonum.org/v1/gonum/mat"
)

type Model struct {
	Layers                []any
	TrainableLayers       []*layer.Layer
	Lossfn                loss.ILoss
	Optimizer             optimization.IOptimizer
	InputLayer            *layer.Input
	OutputLayerActivation activation.IActivation
	Accuracy              accuracy.IAccuracy
}

func New() *Model {
	return &Model{
		Layers: []any{},
	}
}

func (m *Model) Add(layer layer.ILayer) {
	m.Layers = append(m.Layers, layer)
}

func (m *Model) Set(lossfn loss.ILoss, optimizer optimization.IOptimizer, accuracy accuracy.IAccuracy) {
	m.Lossfn = lossfn
	m.Optimizer = optimizer
	m.Accuracy = accuracy
}

func (m *Model) Train(X, y *mat.Dense, epochs int, print_every int) {
	m.Accuracy.Init(y, false)

	for epoch := 0; epoch < epochs+1; epoch++ {
		output := m.forward(X)

		data_loss, regularization_loss := m.Lossfn.Calculate(output, y)
		loss_value := data_loss + regularization_loss

		predictions := m.OutputLayerActivation.Predictions(output)
		accuracy_ := accuracy.Calculate(m.Accuracy, predictions, y)

		m.Backward(output, y)

		m.Optimizer.PreUpdateParams()
		for i := 0; i < len(m.TrainableLayers); i++ {
			m.Optimizer.UpdateParams(m.TrainableLayers[i])
		}
		m.Optimizer.PostUpdateParams()

		if epoch%print_every == 0 {
			fmt.Printf("epoch: %d, acc: %.3f, loss: %.3f, data-loss: %.3f, rg-loss: %.3f, lr: %f\n", epoch, accuracy_, loss_value, data_loss, regularization_loss, m.Optimizer.GetCurrentLearningRate())
		}
	}
}

func (m *Model) forward(X *mat.Dense) *mat.Dense {
	m.InputLayer.Forward(X)

	var output *mat.Dense
	for i := 0; i < len(m.Layers); i++ {
		// since only the prev layer is called it shouldn't affect the loss func as the next layer to the last layer
		m.Layers[i].(layer.ILayer).Forward(m.Layers[i].(layer.ILayer).GetPreviousLayer().(layer.ILayer).GetOutput())
		output = m.Layers[i].(layer.ILayer).GetOutput()
	}

	return output
}

func (m *Model) Backward(output, y *mat.Dense) {
	m.Lossfn.Backward(output, y)

	for i := len(m.Layers) - 1; i >= 0; i-- {
		if i == len(m.Layers)-1 {
			m.Layers[i].(layer.ILayer).Backward(m.Lossfn.GetDInputs())
		} else {
			m.Layers[i].(layer.ILayer).Backward(m.Layers[i].(layer.ILayer).GetNextLayer().(layer.ILayer).GetDInputs())
		}
	}
}

func (m *Model) Finalize() {
	m.InputLayer = new(layer.Input)
	layers_count := len(m.Layers)

	m.TrainableLayers = []*layer.Layer{}

	for i := 0; i < layers_count; i++ {
		// first layer
		if i == 0 {
			if layer, ok := m.Layers[i].(layer.ILayer); ok {
				layer.SetPreviousLayer(m.InputLayer)
				layer.SetNextLayer(m.Layers[i+1])
				// if it doesn't work try overwriting the layer
			}
		} else if i < layers_count-1 /** all layers asides 1st and last**/ {
			if layer, ok := m.Layers[i].(layer.ILayer); ok {
				layer.SetPreviousLayer(m.Layers[i-1])
				layer.SetNextLayer(m.Layers[i+1])
			}
		} else {
			if layer, ok := m.Layers[i].(layer.ILayer); ok {
				layer.SetPreviousLayer(m.Layers[i-1])
				layer.SetNextLayer(m.Lossfn)

				m.OutputLayerActivation = layer.(activation.IActivation)
			}
		}

		// since only layers have Weights
		if layer, ok := m.Layers[i].(*layer.Layer); ok {
			m.TrainableLayers = append(m.TrainableLayers, layer)
		}
	}

	m.Lossfn.RememberTrainableLayers(m.TrainableLayers)
}

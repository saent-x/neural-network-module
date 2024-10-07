package loss

import (
	"github.com/saent-x/ids-nn/core/layer"
	"gonum.org/v1/gonum/mat"
	"math"
)

type ILoss interface {
	Calculate(output *mat.Dense, y *mat.Dense, include_regularization bool) (float64, float64)
	Forward(y_pred *mat.Dense, y_true *mat.Dense) *mat.VecDense
	Backward(d_values *mat.Dense, y_true *mat.Dense)
	CalcRegularizationLoss() float64
	//CalcRegularizationLoss(layer *layer.Layer) float64
	RememberTrainableLayers(trainable_layers []*layer.Layer)
	GetDInputs() *mat.Dense
	SetDInputs(inputs *mat.Dense)
	NewPass()
	CalculateAccumulated(include_regularization bool) (float64, float64)

	layer.ILayerNavigation
}
type Loss struct {
	Regularization_Loss float64
	TrainableLayers     []*layer.Layer
	AccumulatedSum      float64
	AccumulatedCount    float64

	layer.LayerCommons
	layer.LayerNavigation
}

func (loss *Loss) RememberTrainableLayers(trainable_layers []*layer.Layer) {
	loss.TrainableLayers = trainable_layers
}

func (loss *Loss) CalculateAccumulated(include_regularization bool) (float64, float64) {
	dataLoss := loss.AccumulatedSum / loss.AccumulatedCount

	if !include_regularization {
		return dataLoss, 0
	}

	return dataLoss, loss.CalcRegularizationLoss()
}

func (loss *Loss) NewPass() {
	loss.AccumulatedSum = 0
	loss.AccumulatedCount = 0
}

func (loss *Loss) CalcRegularizationLoss() float64 {
	loss.Regularization_Loss = 0

	for i := 0; i < len(loss.TrainableLayers); i++ {
		layer := loss.TrainableLayers[i]
		var abs_weights, abs_biases mat.Dense

		abs_weights.Apply(func(i, j int, v float64) float64 {
			return math.Abs(v)
		}, layer.Weights)
		abs_biases.Apply(func(i, j int, v float64) float64 {
			return math.Abs(v)
		}, layer.Biases)

		if layer.Weight_Regularizer_L1 > 0 {
			loss.Regularization_Loss += layer.Weight_Regularizer_L1 * mat.Sum(&abs_weights)
		}

		if layer.Weight_Regularizer_L2 > 0 {
			var weights_by_weights mat.Dense

			weights_by_weights.MulElem(layer.Weights, layer.Weights)
			loss.Regularization_Loss += layer.Weight_Regularizer_L2 * mat.Sum(&weights_by_weights)
		}

		if layer.Biases_Regularizer_L1 > 0 {
			loss.Regularization_Loss += layer.Biases_Regularizer_L1 * mat.Sum(&abs_biases)
		}

		if layer.Biases_Regularizer_L2 > 0 {
			var biases_by_biases mat.Dense

			biases_by_biases.MulElem(layer.Biases, layer.Biases)
			loss.Regularization_Loss += layer.Biases_Regularizer_L2 * mat.Sum(&biases_by_biases)
		}
	}

	return loss.Regularization_Loss
}

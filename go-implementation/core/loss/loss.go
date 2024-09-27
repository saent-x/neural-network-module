package loss

import (
	"github.com/saent-x/ids-nn/core/layer"
	"gonum.org/v1/gonum/mat"
	"math"
)

type ILoss interface {
	Calculate(output *mat.Dense, y *mat.Dense) (float64, float64)
	Forward(y_pred *mat.Dense, y_true *mat.Dense) *mat.VecDense
	Backward(d_values *mat.Dense, y_true *mat.Dense)
	CalcRegularizationLoss() float64
	//CalcRegularizationLoss(layer *layer.Layer) float64
	RememberTrainableLayers(trainable_layers []*layer.Layer)
	GetDInputs() *mat.Dense

	layer.ILayerNavigation
}
type Loss struct {
	Regularization_Loss float64
	TrainableLayers     []*layer.Layer

	layer.LayerCommons
	layer.LayerNavigation
}

func (l *Loss) RememberTrainableLayers(trainable_layers []*layer.Layer) {
	l.TrainableLayers = trainable_layers
}

func (l *Loss) CalcRegularizationLoss() float64 {
	l.Regularization_Loss = 0

	for i := 0; i < len(l.TrainableLayers); i++ {
		layer := l.TrainableLayers[i]
		var abs_weights, abs_biases mat.Dense

		abs_weights.Apply(func(i, j int, v float64) float64 {
			return math.Abs(v)
		}, layer.Weights)
		abs_biases.Apply(func(i, j int, v float64) float64 {
			return math.Abs(v)
		}, layer.Biases)

		if layer.Weight_Regularizer_L1 > 0 {
			l.Regularization_Loss += layer.Weight_Regularizer_L1 * mat.Sum(&abs_weights)
		}

		if layer.Weight_Regularizer_L2 > 0 {
			var weights_by_weights mat.Dense

			weights_by_weights.MulElem(layer.Weights, layer.Weights)
			l.Regularization_Loss += layer.Weight_Regularizer_L2 * mat.Sum(&weights_by_weights)
		}

		if layer.Biases_Regularizer_L1 > 0 {
			l.Regularization_Loss += layer.Biases_Regularizer_L1 * mat.Sum(&abs_biases)
		}

		if layer.Biases_Regularizer_L2 > 0 {
			var biases_by_biases mat.Dense

			biases_by_biases.MulElem(layer.Biases, layer.Biases)
			l.Regularization_Loss += layer.Biases_Regularizer_L2 * mat.Sum(&biases_by_biases)
		}
	}

	return l.Regularization_Loss
}

// TODO: to be removed
func (l *Loss) CalcRegularizationLoss_1(layer *layer.Layer) float64 {
	l.Regularization_Loss = 0

	var abs_weights, abs_biases mat.Dense

	abs_weights.Apply(func(i, j int, v float64) float64 {
		return math.Abs(v)
	}, layer.Weights)
	abs_biases.Apply(func(i, j int, v float64) float64 {
		return math.Abs(v)
	}, layer.Biases)

	if layer.Weight_Regularizer_L1 > 0 {
		l.Regularization_Loss += layer.Weight_Regularizer_L1 * mat.Sum(&abs_weights)
	}

	if layer.Weight_Regularizer_L2 > 0 {
		var weights_by_weights mat.Dense

		weights_by_weights.MulElem(layer.Weights, layer.Weights)
		l.Regularization_Loss += layer.Weight_Regularizer_L2 * mat.Sum(&weights_by_weights)
	}

	if layer.Biases_Regularizer_L1 > 0 {
		l.Regularization_Loss += layer.Biases_Regularizer_L1 * mat.Sum(&abs_biases)
	}

	if layer.Biases_Regularizer_L2 > 0 {
		var biases_by_biases mat.Dense

		biases_by_biases.MulElem(layer.Biases, layer.Biases)
		l.Regularization_Loss += layer.Biases_Regularizer_L2 * mat.Sum(&biases_by_biases)
	}

	return l.Regularization_Loss
}

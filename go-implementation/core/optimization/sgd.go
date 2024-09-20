package optimization

import (
	"github.com/saent-x/ids-nn/core/layer"
	"gonum.org/v1/gonum/mat"
)

type StochasticGradientDescent struct {
	LearningRate        float64
	CurrentLearningRate float64
	Decay               float64
	Iterations          float64
	Momentum            float64
}

func CreateStochasticGradientDescent(learningRate float64, decay float64, momentum float64) *StochasticGradientDescent {
	return &StochasticGradientDescent{
		LearningRate:        learningRate,
		CurrentLearningRate: learningRate,
		Decay:               decay,
		Iterations:          0.0,
		Momentum:            momentum,
	}
}

func (self *StochasticGradientDescent) PreUpdateParams() {
	if self.Decay != 0 {
		self.CurrentLearningRate = self.LearningRate * (1. / (1. + self.Decay*self.Iterations))
	}
}

func (self *StochasticGradientDescent) UpdateParams(layer *layer.Layer) {
	r, c := layer.D_Weights.Dims()
	r0, c0 := layer.D_Biases.Dims()

	new_weights := mat.NewDense(r, c, nil)
	new_biases := mat.NewDense(r0, c0, nil)

	if self.Momentum > 0 {
		if layer.Weights_Momentum == nil || layer.Biases_Momentum == nil {
			layer.Weights_Momentum = mat.DenseCopyOf(new_weights)
			layer.Biases_Momentum = mat.DenseCopyOf(new_biases)

			layer.Weights_Momentum.Zero()
			layer.Biases_Momentum.Zero()
		}

		var weight_momentum_mul, bias_momentum_mul mat.Dense
		var dweights_lr_mul, dbiases_lr_mul mat.Dense

		weight_momentum_mul.Apply(func(i, j int, v float64) float64 {
			return self.Momentum * v
		}, layer.Weights_Momentum)

		dweights_lr_mul.Apply(func(i, j int, v float64) float64 {
			return self.CurrentLearningRate * v
		}, layer.D_Weights)

		new_weights.Sub(&weight_momentum_mul, &dweights_lr_mul)

		bias_momentum_mul.Apply(func(i, j int, v float64) float64 {
			return self.Momentum * v
		}, layer.Biases_Momentum)

		dbiases_lr_mul.Apply(func(i, j int, v float64) float64 {
			return self.CurrentLearningRate * v
		}, layer.D_Biases)

		new_biases.Sub(&bias_momentum_mul, &dbiases_lr_mul)

		layer.Weights_Momentum = mat.DenseCopyOf(new_weights)
		layer.Biases_Momentum = mat.DenseCopyOf(new_biases)

	} else {
		// multiply by the negative of the learning rate
		new_weights.Apply(func(i, j int, v float64) float64 {
			return -self.CurrentLearningRate * v
		}, layer.D_Weights)

		new_biases.Apply(func(i, j int, v float64) float64 {
			return -self.CurrentLearningRate * v
		}, layer.D_Biases)
	}

	var weights_sum, biases_sum mat.Dense

	weights_sum.Add(layer.Weights, new_weights)
	biases_sum.Add(layer.Biases, new_biases)

	layer.Weights = mat.DenseCopyOf(&weights_sum)
	layer.Biases = mat.DenseCopyOf(&biases_sum)
}

func (self *StochasticGradientDescent) PostUpdateParams() {
	self.Iterations += 1
}

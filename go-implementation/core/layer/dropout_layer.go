package layer

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type DropoutLayer struct {
	Rate       float64
	BinaryMask *mat.Dense

	LayerCommons
	LayerNavigation
}

func NewDropoutLayer(rate float64) *DropoutLayer {
	return &DropoutLayer{
		Rate: 1 - rate,
	}
}

func (dropoutLayer *DropoutLayer) Forward(inputs *mat.Dense, training bool) {
	dropoutLayer.Inputs = mat.DenseCopyOf(inputs)

	if !training {
		dropoutLayer.Output = mat.DenseCopyOf(inputs)
		return
	}

	rows, cols := inputs.Dims()
	dropoutLayer.BinaryMask = mat.NewDense(rows, cols, nil)

	binomial := distuv.Binomial{N: 1, P: dropoutLayer.Rate}

	dropoutLayer.BinaryMask.Apply(func(i, j int, v float64) float64 {
		// Generate binary value using binomial distribution (either 0 or 1)
		maskValue := binomial.Rand()
		if maskValue == 1 {
			return 1.0 / dropoutLayer.Rate // Scale mask by 1 / rate
		} else {
			return 0
		}
	}, dropoutLayer.BinaryMask)

	// Apply mask to inputs (element-wise multiplication)
	dropoutLayer.Output = mat.NewDense(rows, cols, nil)
	dropoutLayer.Output.MulElem(inputs, dropoutLayer.BinaryMask)
}

func (dropoutLayer *DropoutLayer) Backward(d_values *mat.Dense) {
	var new_dinputs mat.Dense
	new_dinputs.MulElem(d_values, dropoutLayer.BinaryMask)

	dropoutLayer.D_Inputs = mat.DenseCopyOf(&new_dinputs)
}

package layer

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type DropoutLayer struct {
	Rate       float64
	Inputs     *mat.Dense
	BinaryMask *mat.Dense
	
	LayerCommons

	prev interface{}
	next interface{}
}

func NewDropoutLayer(rate float64) *DropoutLayer {
	return &DropoutLayer{
		Rate: 1 - rate,
	}
}

func (l *DropoutLayer) Forward(inputs *mat.Dense) {
	l.Inputs = mat.DenseCopyOf(inputs)

	rows, cols := inputs.Dims()
	l.BinaryMask = mat.NewDense(rows, cols, nil)

	binomial := distuv.Binomial{N: 1, P: l.Rate}

	l.BinaryMask.Apply(func(i, j int, v float64) float64 {
		// Generate binary value using binomial distribution (either 0 or 1)
		maskValue := binomial.Rand()
		if maskValue == 1 {
			return 1.0 / l.Rate // Scale mask by 1 / rate
		} else {
			return 0
		}
	}, l.BinaryMask)

	// Apply mask to inputs (element-wise multiplication)
	l.Output = mat.NewDense(rows, cols, nil)
	l.Output.MulElem(inputs, l.BinaryMask)
}

func (l *DropoutLayer) Backward(d_values *mat.Dense) {
	var new_dinputs mat.Dense
	new_dinputs.MulElem(d_values, l.BinaryMask)

	l.D_Inputs = mat.DenseCopyOf(&new_dinputs)
}

func (self *DropoutLayer) GetPreviousLayer() interface{} {
	return self.prev
}

func (self *DropoutLayer) GetNextLayer() interface{} {
	return self.next
}

func (self *DropoutLayer) SetPreviousLayer(prev interface{}) {
	self.prev = prev
}

func (self *DropoutLayer) SetNextLayer(next interface{}) {
	self.next = next
}

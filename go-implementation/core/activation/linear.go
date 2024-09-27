package activation

import (
	"github.com/saent-x/ids-nn/core/layer"
	"gonum.org/v1/gonum/mat"
)

type Linear struct {
	Inputs *mat.Dense

	layer.LayerCommons
	layer.LayerNavigation
}

func (l *Linear) Forward(inputs *mat.Dense) {
	l.Inputs = inputs
	l.Output = inputs
}

func (l *Linear) Backward(d_values *mat.Dense) {
	l.D_Inputs = mat.DenseCopyOf(d_values)
}

func (l *Linear) GetOutput() *mat.Dense {
	return l.Output
}

func (l *Linear) Predictions(outputs *mat.Dense) *mat.Dense {
	return outputs
}

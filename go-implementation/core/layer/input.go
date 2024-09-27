package layer

import "gonum.org/v1/gonum/mat"

type Input struct {
	LayerCommons
	LayerNavigation
}

func (i *Input) Forward(inputs *mat.Dense) {
	i.Output = mat.DenseCopyOf(inputs)
}

// [Redundant function]: only exists to satisfy interface constraint
func (self *Input) Backward(d_values *mat.Dense) {}
func (self *Input) GetOutput() *mat.Dense {
	return self.Output
}

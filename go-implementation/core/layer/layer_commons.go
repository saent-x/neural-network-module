package layer

import "gonum.org/v1/gonum/mat"

type LayerCommons struct {
	D_Inputs *mat.Dense
	Output   *mat.Dense
}

func (lc *LayerCommons) GetDInputs() *mat.Dense {
	return lc.D_Inputs
}

func (lc *LayerCommons) GetOutput() *mat.Dense {
	return lc.Output
}

package layer

import "gonum.org/v1/gonum/mat"

type LayerCommons struct {
	Inputs   *mat.Dense
	D_Inputs *mat.Dense
	Output   *mat.Dense
}

func (layerCommons *LayerCommons) GetDInputs() *mat.Dense {
	return layerCommons.D_Inputs
}

func (layerCommons *LayerCommons) SetDInputs(inputs *mat.Dense) {
	if inputs == nil {
		layerCommons.D_Inputs = nil
	} else {
		layerCommons.D_Inputs = mat.DenseCopyOf(inputs)
	}
}

func (layerCommons *LayerCommons) GetOutput() *mat.Dense {
	return layerCommons.Output
}

func (layerCommons *LayerCommons) Reset() {
	layerCommons.D_Inputs = nil
	layerCommons.Output = nil
	layerCommons.Inputs = nil
}

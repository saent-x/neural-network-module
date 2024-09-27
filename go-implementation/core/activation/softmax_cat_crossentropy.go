package activation

import (
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
)

type SoftmaxCatCrossEntropy struct {
	Activation *SoftMax
	CE_Loss    *loss.CrossEntropyLossFunction
	Output     *mat.Dense

	loss.Loss
	layer.LayerNavigation
}

func CreateSoftmaxCatCrossEntropy() *SoftmaxCatCrossEntropy {
	self := new(SoftmaxCatCrossEntropy)

	self.Activation = new(SoftMax)
	self.CE_Loss = new(loss.CrossEntropyLossFunction)

	return self
}

func (self *SoftmaxCatCrossEntropy) Calculate(inputs *mat.Dense, y_true *mat.Dense) (float64, float64) {
	self.Activation.Forward(inputs)
	self.Output = self.Activation.Output

	return self.CE_Loss.Calculate(self.Output, y_true)
}

func (self *SoftmaxCatCrossEntropy) Backward(d_values *mat.Dense, y_true *mat.Dense) {
	samples, _ := d_values.Dims()
	rows, _ := y_true.Dims()

	if rows > 1 {
		// find index of max value in each row - convert OHE to sparse values
		var y_true_new mat.VecDense
		for i := 0; i < rows; i++ {
			row := y_true.RawRowView(i)
			arg_max := lo.IndexOf(row, lo.Max(row))

			y_true_new.SetVec(i, float64(arg_max))
		}

		y_true = mat.NewDense(y_true_new.Len(), 1, y_true_new.RawVector().Data)
	}

	self.D_Inputs = mat.DenseCopyOf(d_values)

	// calculate gradient
	for i := 0; i < samples; i++ {
		y_true_raw := y_true.RawMatrix().Data
		self.D_Inputs.Set(i, int(y_true_raw[i]), self.D_Inputs.At(i, int(y_true_raw[i]))-1)
	}

	self.D_Inputs.Apply(func(i, j int, v float64) float64 {
		return v / float64(samples)
	}, self.D_Inputs)
}

func (self *SoftmaxCatCrossEntropy) GetOutput() *mat.Dense {
	return self.Output
}

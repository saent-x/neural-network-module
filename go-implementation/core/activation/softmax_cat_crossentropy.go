package activation

import (
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
	"math"
)

type SoftmaxCatCrossEntropy struct {
	Activation          *SoftMax
	Loss                *loss.CrossEntropyLossFunction
	Output              *mat.Dense
	D_Inputs            *mat.Dense
	Regularization_Loss float64
}

func CreateSoftmaxCatCrossEntropy() *SoftmaxCatCrossEntropy {
	self := new(SoftmaxCatCrossEntropy)

	self.Activation = new(SoftMax)
	self.Loss = new(loss.CrossEntropyLossFunction)

	return self
}

func (self *SoftmaxCatCrossEntropy) Forward(inputs *mat.Dense, y_true *mat.Dense) float64 {
	self.Activation.Forward(inputs)
	self.Output = self.Activation.Output

	return self.Loss.Calculate(self.Output, y_true)
}

func (self *SoftmaxCatCrossEntropy) Backward(d_values *mat.Dense, y_true *mat.Dense) {
	samples, _ := d_values.Dims()
	rows, _ := y_true.Dims()

	if rows > 1 {
		// find index of max value in each row - convert OHE to sparse values
		y_true_new := mat.NewVecDense(rows, nil)
		lo.ForEach(lo.Range(rows), func(item int, index int) {
			row := y_true.RawRowView(index)
			arg_max := lo.IndexOf(row, lo.Max(row))

			y_true_new.SetVec(index, float64(arg_max))
		})

		y_true = mat.NewDense(y_true_new.Len(), 1, y_true_new.RawVector().Data)
	}

	self.D_Inputs = mat.DenseCopyOf(d_values)

	// calculate gradient
	lo.ForEach(lo.Range(samples), func(item int, i int) {
		y_true_raw := y_true.RawMatrix().Data
		self.D_Inputs.Set(i, int(y_true_raw[i]), self.D_Inputs.At(i, int(y_true_raw[i]))-1)
	})

	self.D_Inputs.Apply(func(i, j int, v float64) float64 {
		return v / float64(samples)
	}, self.D_Inputs)
}

func (lf *SoftmaxCatCrossEntropy) CalcRegularizationLoss(layer *layer.Layer) float64 {
	lf.Regularization_Loss = 0

	var abs_weights, abs_biases mat.Dense

	abs_weights.Apply(func(i, j int, v float64) float64 {
		return math.Abs(v)
	}, layer.Weights)
	abs_biases.Apply(func(i, j int, v float64) float64 {
		return math.Abs(v)
	}, layer.Weights)

	if layer.Weight_Regularizer_L1 > 0 {
		lf.Regularization_Loss += layer.Weight_Regularizer_L1 * mat.Sum(&abs_weights)
	}

	if layer.Weight_Regularizer_L2 > 0 {
		var weights_by_weights mat.Dense

		weights_by_weights.MulElem(layer.Weights, layer.Weights)
		lf.Regularization_Loss += layer.Weight_Regularizer_L2 * mat.Sum(&weights_by_weights)
	}

	if layer.Biases_Regularizer_L1 > 0 {
		lf.Regularization_Loss += layer.Biases_Regularizer_L1 * mat.Sum(&abs_biases)
	}

	if layer.Biases_Regularizer_L2 > 0 {
		var biases_by_biases mat.Dense

		biases_by_biases.MulElem(layer.Biases, layer.Biases)
		lf.Regularization_Loss += layer.Biases_Regularizer_L2 * mat.Sum(&biases_by_biases)
	}

	return lf.Regularization_Loss
}

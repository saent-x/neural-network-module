package optimization

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot/plotter"
	"image/color"
	"log"
	"math"
	"math/rand/v2"
	"testing"
)

func TestForNNOptimization_1(t *testing.T) {
	X, y := core.SpiralData(100, 3)

	layer_1 := layer.CreateLayer(2, 3, 0, 0, 0, 0)
	layer_2 := layer.CreateLayer(3, 3, 0, 0, 0, 0)

	activation_1 := new(activation.ReLU)
	activation_2 := new(activation.SoftMax) // for the output layer

	lossfn_1 := new(loss.CrossEntropyLossFunction)

	layer_1.Forward(X)
	activation_1.Forward(layer_1.Output)

	layer_2.Forward(activation_1.Output)
	activation_2.Forward(layer_2.Output)

	// initial variables for optimizer
	lowest_loss := 9999999.0
	best_layer_weight_1 := layer_1.Weights
	best_layer_bias_1 := layer_1.Biases
	best_layer_weight_2 := layer_2.Weights
	best_layer_bias_2 := layer_2.Biases

	for _, ix := range lo.Range(100000) {
		// generate new sets of weights per iteration
		layer_1.Weights.Apply(generateRand, layer_1.Weights)
		layer_1.Biases.Apply(generateRand, layer_1.Biases)

		layer_2.Weights.Apply(generateRand, layer_2.Weights)
		layer_2.Biases.Apply(generateRand, layer_2.Biases)

		layer_1.Forward(X)
		activation_1.Forward(layer_1.Output)

		layer_2.Forward(activation_1.Output)
		activation_2.Forward(layer_2.Output)

		loss_value := lossfn_1.Calculate(activation_2.Output, y)
		accuracy_ := accuracy.Calculate(activation_2.Output, y)

		if loss_value < lowest_loss {
			fmt.Println("New set of weights found, iteration: ", ix, "loss: ", loss_value, "accuracy: ", accuracy_)

			best_layer_weight_1 = layer_1.Weights
			best_layer_bias_1 = layer_1.Biases
			best_layer_weight_2 = layer_2.Weights
			best_layer_bias_2 = layer_2.Biases

			lowest_loss = loss_value
		} else {
			layer_1.Weights = mat.DenseCopyOf(best_layer_weight_1)
			layer_1.Biases = mat.DenseCopyOf(best_layer_bias_1)
			layer_2.Weights = mat.DenseCopyOf(best_layer_weight_2)
			layer_2.Biases = mat.DenseCopyOf(best_layer_bias_2)
		}

	}
}

func generateRand(i int, j int, v float64) float64 {
	return v + 0.05*rand.NormFloat64()
}

func fn(x []float64) []float64 {
	result := make([]float64, len(x))
	lo.ForEach(x, func(item float64, index int) {
		result[index] = 2 * math.Pow(item, 2)
	})
	return result
}

func f(x float64) float64 {
	return 2 * math.Pow(x, 2)
}

func tangentLine(x float64, approximate_derivative float64, b float64) float64 {
	return approximate_derivative*x + b
}

func TestDerivativeCalculation_1(t *testing.T) {
	x := lo.RangeWithSteps(0, 5, 0.001)
	y := fn(x)

	p := core.PlotLine(x, y)

	p2_delta := 0.0001
	x1 := 2.0
	x2 := x1 + p2_delta

	y1 := f(x1)
	y2 := f(x2)

	fmt.Printf("(%f, %f) (%f, %f)\n", x1, y1, x2, y2)

	// Derivative approximation and y-intercept for the tangent line
	approximate_derivative := (y2 - y1) / (x2 - x1)
	b := y2 - approximate_derivative*2

	to_plot := []float64{x1 - 0.9, x1, x1 + 0.9}
	y_tangent_line := make([]float64, len(to_plot))
	lo.ForEach(to_plot, func(item float64, index int) {
		y_tangent_line[index] = tangentLine(item, approximate_derivative, b)
	})

	XY_pts := make(plotter.XYs, len(y_tangent_line))

	lo.ForEach(XY_pts, func(_ plotter.XY, index int) {
		XY_pts[index].X = to_plot[index]
		XY_pts[index].Y = y_tangent_line[index]
	})

	nl, err := plotter.NewLine(XY_pts)
	if err != nil {
		log.Fatalf("could not create line plot: %v", err)
	}

	nl.Color = color.RGBA{R: 0, G: 255, B: 0, A: 255}
	p.Add(nl)

	result := core.SavePlot(p, "derivative_plot")

	if !result {
		t.Errorf("error: no plot was made")
	}

	fmt.Printf("Approximate derivative for f(x) where x = %f is %f\n", x1, approximate_derivative)
}

func TestBackPropagation_1(t *testing.T) {
	x := mat.NewVecDense(3, []float64{1.0, -2.0, 3.0})
	w := mat.NewVecDense(3, []float64{-3.0, -1.0, 2.0})
	b := 1.0

	xw0 := x.AtVec(0) * w.AtVec(0)
	xw1 := x.AtVec(1) * w.AtVec(1)
	xw2 := x.AtVec(2) * w.AtVec(2)

	z := xw0 + xw1 + xw2 + b

	d_value := 1.0 // deriv from next layer
	d_relu_dz := d_value * (lo.Ternary[float64](z > 0, 1, 0))

	drelu_dxw0 := d_relu_dz * 1.0
	drelu_dxw1 := d_relu_dz * 1.0
	drelu_dxw2 := d_relu_dz * 1.0
	drelu_db := d_relu_dz * 1.0

	// partial derivatives of the mul, chain rule

	drelu_dx0 := drelu_dxw0 * w.AtVec(0)
	drelu_dx1 := drelu_dxw1 * w.AtVec(1)
	drelu_dx2 := drelu_dxw2 * w.AtVec(2)
	drelu_dxw0 = drelu_dxw0 * x.AtVec(0)
	drelu_dxw1 = drelu_dxw1 * x.AtVec(1)
	drelu_dxw2 = drelu_dxw2 * x.AtVec(2)

	_ = []float64{drelu_dx0, drelu_dx1, drelu_dx2}
	dw := []float64{drelu_dxw0, drelu_dxw1, drelu_dxw2}
	db := drelu_db

	// apply fraction of gradient
	w.SetVec(0, w.AtVec(0)+(-0.001*dw[0]))
	w.SetVec(1, w.AtVec(1)+(-0.001*dw[1]))
	w.SetVec(2, w.AtVec(2)+(-0.001*dw[2]))
	b += -0.001 * db

	// multiply inputs by weights
	xw0 = x.AtVec(0) * w.AtVec(0)
	xw1 = x.AtVec(1) * w.AtVec(1)
	xw2 = x.AtVec(2) * w.AtVec(2)

	z = xw0 + xw1 + xw2 + b
	y := math.Max(z, 0)

	fmt.Println(w, b, y)
}

func TestBackPropagation_2(t *testing.T) {
	X := mat.NewDense(4, 3, []float64{0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08, 0.02, 0.9, 0.08})
	d_values := mat.NewDense(3, 3, []float64{1., 1., 1., 2., 2., 2., 3., 3., 3.})

	layer_1 := layer.CreateLayer(3, 3, 0, 0, 0, 0)
	layer_1.Forward(X)

	layer_1.Backward(d_values)

	fmt.Println(mat.Formatted(layer_1.D_Weights))
}

func TestLearningRateDecay(t *testing.T) {
	start_learning_rate := 1.
	learning_rate_decay := 0.1

	var learning_rate float64
	for step, _ := range lo.Range(20) {
		learning_rate = start_learning_rate * (1. / (1 + learning_rate_decay*float64(step)))
		fmt.Println(learning_rate)
	}
}

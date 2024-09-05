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

	layer_1 := layer.CreateLayer(2, 3)
	layer_2 := layer.CreateLayer(3, 3)

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

func TestSimple(t *testing.T) {
	a := []float64{1, 2, 3}

	lo.ForEach(a, func(item float64, index int) {
		item += 1
	})

	lo.ForEach(a, func(item float64, index int) {
		fmt.Println(item)
	})
}

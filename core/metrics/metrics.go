package metrics

import (
	"fmt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// ConfusionMatrix Function to calculate confusion matrix for both binary and multi-class problems
func ConfusionMatrix(yTrue, yPred []float64, numClasses int) [][]float64 {
	matrix := make([][]float64, numClasses)
	for i := range matrix {
		matrix[i] = make([]float64, numClasses)
	}

	for i := 0; i < len(yTrue); i++ {
		matrix[int(yTrue[i])][int(yPred[i])]++
	}

	return matrix
}

func CalculateMetrics(confusionMatrix [][]float64, numClasses int) (accuracy, precision, recall, f1Score []float64) {
	total := 0.
	correct := 0.
	accuracy = make([]float64, 1)
	precision = make([]float64, numClasses)
	recall = make([]float64, numClasses)
	f1Score = make([]float64, numClasses)

	for i := 0; i < numClasses; i++ {
		truePositive := confusionMatrix[i][i]
		falsePositive := 0.
		falseNegative := 0.
		totalClass := 0.

		for j := 0; j < numClasses; j++ {
			totalClass += confusionMatrix[i][j] // Total for class i
			total += confusionMatrix[i][j]
			if i != j {
				falsePositive += confusionMatrix[j][i]
				falseNegative += confusionMatrix[i][j]
			}
		}

		if truePositive+falsePositive > 0 {
			precision[i] = float64(truePositive) / float64(truePositive+falsePositive)
		}
		if truePositive+falseNegative > 0 {
			recall[i] = float64(truePositive) / float64(truePositive+falseNegative)
		}
		if precision[i]+recall[i] > 0 {
			f1Score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
		}

		correct += truePositive
	}

	accuracy[0] = correct / total
	return
}

// ConfusionMatrixHeatMap holds the confusion matrix data and implements the GridHeatMap interface
type ConfusionMatrixHeatMap struct {
	Matrix     [][]float64
	NumClasses int
}

// Dims returns the number of rows and columns in the confusion matrix
func (h ConfusionMatrixHeatMap) Dims() (c, r int) {
	return h.NumClasses, h.NumClasses
}

// Z returns the value at the given row and column
func (h ConfusionMatrixHeatMap) Z(c, r int) float64 {
	return float64(h.Matrix[h.NumClasses-r-1][c]) // Invert Y axis
}

// X returns the X value for the given column (just the class index)
func (h ConfusionMatrixHeatMap) X(c int) float64 {
	return float64(c)
}

// Y returns the Y value for the given row (just the class index, inverted)
func (h ConfusionMatrixHeatMap) Y(r int) float64 {
	return float64(r)
}

// LabelValue is used to plot text labels on each heatmap cell
type LabelValue struct {
	XVal, YVal float64
	Label      string
}

// XYs returns the coordinates for the labels
func (l LabelValue) XYs() plotter.XYs {
	return plotter.XYs{{X: l.XVal, Y: l.YVal}}
}

// Labels implements plotter.Labels to provide custom labels for the plot
type Labels struct {
	Values []LabelValue
}

// XY returns the coordinates for the ith label
func (l Labels) XY(i int) (float64, float64) {
	return l.Values[i].XVal, l.Values[i].YVal
}

// Label returns the string label for the ith label
func (l Labels) Label(i int) string {
	return l.Values[i].Label
}

// Len returns the number of labels
func (l Labels) Len() int {
	return len(l.Values)
}

// PlotConfusionMatrix Function to plot confusion matrix as a heatmap with dynamic tick markers based on the number of classes
func PlotConfusionMatrix(matrix [][]float64, numClasses int, filename string) {
	p := plot.New()

	p.Title.Text = "Confusion Matrix"
	p.X.Label.Text = "Predicted Class"
	p.Y.Label.Text = "True Class"

	// Create a heatmap based on the confusion matrix
	heatmap := ConfusionMatrixHeatMap{
		Matrix:     matrix,
		NumClasses: numClasses,
	}

	// Create a color map palette and heatmap plotter
	pal := moreland.SmoothBlueRed().Palette(255)
	hm := plotter.NewHeatMap(heatmap, pal)

	// Add the heatmap to the plot
	p.Add(hm)

	// Dynamically set tick markers for X and Y axes based on number of classes
	xTicks := make([]plot.Tick, numClasses)
	yTicks := make([]plot.Tick, numClasses)

	for i := 0; i < numClasses; i++ {
		label := fmt.Sprintf("%d", i)
		xTicks[i] = plot.Tick{Value: float64(i), Label: label}
		yTicks[i] = plot.Tick{Value: float64(i), Label: label}
	}

	p.X.Tick.Marker = plot.ConstantTicks(xTicks)
	p.Y.Tick.Marker = plot.ConstantTicks(yTicks)

	// Prepare text labels to display values in the confusion matrix
	labels := Labels{}
	for i := 0; i < numClasses; i++ {
		for j := 0; j < numClasses; j++ {
			value := matrix[i][j]
			x := float64(j)
			y := float64(numClasses - i - 1) // Invert Y axis for correct positioning

			// Add a LabelValue for each heatmap cell
			labels.Values = append(labels.Values, LabelValue{
				XVal:  x,
				YVal:  y,
				Label: fmt.Sprintf("%1.f", value),
			})
		}
	}

	l, err1 := plotter.NewLabels(labels)
	if err1 != nil {
		panic(err1)
	}
	// Add the labels to the plot
	p.Add(l)

	// Save the plot as a PNG file
	if err := p.Save(6*vg.Inch, 6*vg.Inch, filename); err != nil {
		panic(err)
	}
}

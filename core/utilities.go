package core

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/saent-x/ids-nn/core/datamodels"
	"github.com/saent-x/ids-nn/core/scaling"
	"github.com/samber/lo"
	"golang.org/x/image/draw"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

func linspace(start, stop float64, num int) []float64 {
	arr := make([]float64, num)
	step := (stop - start) / float64(num-1)
	for i := 0; i < num; i++ {
		arr[i] = start + step*float64(i)
	}
	return arr
}

func SineData(samples int) (*mat.Dense, *mat.Dense) {
	// Create the X matrix (arange equivalent)
	X := mat.NewDense(samples, 1, nil)
	for i := 0; i < samples; i++ {
		X.Set(i, 0, float64(i)/float64(samples))
	}

	// Create the y matrix (sin(2 * pi * X) equivalent)
	Y := mat.NewDense(samples, 1, nil)
	for i := 0; i < samples; i++ {
		xVal := X.At(i, 0)
		Y.Set(i, 0, math.Sin(2*math.Pi*xVal))
	}

	return X, Y
}

func VerticalData(samples, classes int) (*mat.Dense, *mat.Dense) {
	X := mat.NewDense(samples*classes, 2, nil)
	y := make([]float64, samples*classes)

	for _, class_number := range lo.Range(classes) {
		ix := lo.RangeWithSteps(samples*class_number, samples*(class_number+1), 1)

		for _, v := range ix {
			X.Set(v, 0, rand.NormFloat64()*0.1+float64(class_number)/3)
			X.Set(v, 1, rand.NormFloat64()*0.1+0.5)
			y[v] = float64(class_number)
		}
	}
	return X, mat.NewDense(1, len(y), y)
}

func SpiralData(samples, classes int) (*mat.Dense, *mat.Dense) {
	X := mat.NewDense(samples*classes, 2, nil)
	y := make([]float64, samples*classes)

	for classNumber := 0; classNumber < classes; classNumber++ {
		ixStart := samples * classNumber
		ixEnd := samples * (classNumber + 1)

		r := linspace(0.0, 1.0, samples)
		t := linspace(float64(classNumber*4), float64((classNumber+1)*4), samples)
		for i := range t {
			t[i] += rand.NormFloat64() * 0.2 // Adding Gaussian noise
		}

		for i := ixStart; i < ixEnd; i++ {
			rIndex := i - ixStart
			X.Set(i, 0, r[rIndex]*math.Sin(t[rIndex]*2.5))
			X.Set(i, 1, r[rIndex]*math.Cos(t[rIndex]*2.5))
			y[i] = float64(classNumber)
		}
	}

	return X, mat.NewDense(1, len(y), y)
}

func PlotScatter(X *mat.Dense, samples, classes int, filepath string) bool {
	p := plot.New()

	p.Title.Text = fmt.Sprintf("%v Data", lo.PascalCase(filepath))
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	// Define different colors for different classes
	colors := []color.RGBA{
		{R: 255, G: 0, B: 0, A: 255},   // Red
		{R: 0, G: 255, B: 0, A: 255},   // Green
		{R: 0, G: 0, B: 255, A: 255},   // Blue
		{R: 255, G: 255, B: 0, A: 255}, // Yellow
		{R: 255, G: 0, B: 255, A: 255}, // Magenta
	}

	for classNumber := 0; classNumber < classes; classNumber++ {
		pts := make(plotter.XYs, samples)
		for i := 0; i < samples; i++ {
			index := classNumber*samples + i
			pts[i].X = X.At(index, 0) // since Y has two features/ columns
			pts[i].Y = X.At(index, 1)

		}

		s, err := plotter.NewScatter(pts)
		if err != nil {
			log.Fatalf("could not create scatter plot: %v", err)
		}
		s.GlyphStyle.Color = colors[classNumber%len(colors)]
		s.GlyphStyle.Radius = vg.Points(3)

		p.Add(s)
	}

	// Save the plot to a PNG file
	if err := p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%s.png", filepath)); err != nil {
		log.Fatalf("could not save plot: %v", err)

		return false
	} else {
		return true
	}
}

func PlotSineData(X, Y *mat.Dense) bool {
	// Create a new plot
	p := plot.New()

	// Set plot title and labels
	p.Title.Text = "Plot of X vs sin(2πX)"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	// Create XYs which holds the X and Y data
	points := make(plotter.XYs, X.RawMatrix().Rows)
	for i := 0; i < X.RawMatrix().Rows; i++ {
		points[i].X = X.At(i, 0)
		points[i].Y = Y.At(i, 0)
	}

	// Create a scatter plot for the data
	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}

	// Add the scatter plot to the plot
	p.Add(scatter)

	// Save the plot to a PNG file
	if err := p.Save(6*vg.Inch, 4*vg.Inch, "sin_plot.png"); err != nil {
		panic(err)
		return false
	}

	return true
}

func MeanOnLastAxis(matrix *mat.Dense) *mat.VecDense {
	means := mat.NewVecDense(matrix.RawMatrix().Rows, nil)

	for i := 0; i < matrix.RawMatrix().Rows; i++ {
		row := matrix.RawRowView(i)
		means.SetVec(i, stat.Mean(row, nil))
	}

	return means
}

func Sign(v float64) float64 {
	if v < 0 {
		return -1
	} else if v == 0 {
		return 0
	} else if v > 0 {
		return 1
	} else {
		return math.NaN()
	}
}

func PlotLine(x []float64, y []float64) *plot.Plot {
	p := plot.New()

	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	XY_pts := make(plotter.XYs, len(x))

	for i := 0; i < len(XY_pts); i++ {
		XY_pts[i].X = x[i]
		XY_pts[i].Y = y[i]
	}

	s, err := plotter.NewLine(XY_pts)
	if err != nil {
		log.Fatalf("could not create line plot: %v", err)
	}
	s.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255} //red
	p.Add(s)

	return p
}

func SavePlot(p *plot.Plot, filepath string) bool {
	p.Title.Text = fmt.Sprintf("%v Data", lo.PascalCase(filepath))

	// Save the plot to a PNG file
	if err := p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%s.png", filepath)); err != nil {
		log.Fatalf("could not save plot: %v", err)

		return false
	} else {
		return true
	}
}

func SparseToOHE(data *mat.Dense, n int) *mat.Dense {
	_, cols := data.Dims()
	// Create a zero matrix of size n x n
	eyeMatrix := mat.NewDense(n, n, nil)
	outputMatrix := mat.NewDense(cols, n, nil)

	// Set the diagonal elements to 1
	for i := 0; i < n; i++ {
		eyeMatrix.Set(i, i, 1)
	}
	for i := 0; i < cols; i++ {
		outputMatrix.SetRow(i, eyeMatrix.RawRowView(int(data.At(0, i))))
	}

	return outputMatrix
}

func OHEToSparse(data *mat.Dense) *mat.Dense {
	rows, _ := data.Dims()

	result := mat.NewDense(rows, 1, nil)
	for i := 0; i < rows; i++ {
		row_max_idx := floats.MaxIdx(data.RawRowView(i))
		result.Set(i, 0, float64(row_max_idx))
	}

	return result
}

func SumSlices(slice []float64) float64 {
	count := .0
	for _, s := range slice {
		count += s
	}

	return count
}

func FirstN(dense *mat.Dense, n int) *mat.Dense {
	if n > dense.RawMatrix().Rows {
		panic("invalid operation!")
	}
	new_dense := mat.NewDense(n, dense.RawMatrix().Cols, nil)

	for i := 0; i < n; i++ {
		new_dense.SetRow(i, dense.RawRowView(i))
	}

	return new_dense
}

func Fill_n(value float64, n int) []float64 {
	nArr := make([]float64, n)

	for i := 0; i < n; i++ {
		nArr[i] = value
	}

	return nArr
}

func SaveDataToSlice(dirs []os.DirEntry, dirPath string, shuffle bool) (*mat.Dense, *mat.Dense, error) {
	var X [][]float64
	var y []byte

	for _, f := range dirs {
		if f.IsDir() {
			// go through each dir 0 - 9
			imgsPath := fmt.Sprintf("%s/%s", dirPath, f.Name())
			imgs, err := os.ReadDir(imgsPath)

			if err != nil {
				panic(err)
			}

			for _, i := range imgs {
				x_, y_ := readImage(i, imgsPath, f)

				X = append(X, x_)
				y = append(y, y_)
			}
		}
	}

	X_mat := mat.NewDense(len(X), len(X[0]), nil)
	y_mat := mat.NewDense(1, len(y), nil)

	if shuffle {
		shuffledIdxs := ShuffleSlice(GetRange(len(X)))
		for i := 0; i < X_mat.RawMatrix().Rows; i++ {
			idx := shuffledIdxs[i]
			X_mat.SetRow(idx, X[i])
			y_mat.Set(0, idx, float64(y[i]))
		}
	} else {
		for i := 0; i < X_mat.RawMatrix().Rows; i++ {
			X_mat.SetRow(i, X[i])
			y_mat.Set(0, i, float64(y[i]))
		}
	}

	return X_mat, y_mat, nil
}

func ContainsNaN(val *mat.Dense) {
	rows, cols := val.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.IsNaN(val.At(i, j)) {
				fmt.Println("possible culprit!!")
				break
			}
		}
	}
}

func SubtractUnevenMatrices(A, B *mat.Dense) *mat.Dense {
	rowsA, colsA := A.Dims()
	rowsB, colsB := B.Dims()

	if rowsA != rowsB || colsB != 1 {
		panic("Matrix dimensions do not match for broadcasting")
	}

	// Create a result matrix with the same dimensions as A
	result := mat.NewDense(rowsA, colsA, nil)

	// Subtract B from A, broadcasting B across columns
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			// Subtract corresponding element from B (broadcasted across columns)
			result.Set(i, j, A.At(i, j)-B.At(i, 0))
		}
	}

	return result
}

func DivideUnevenMatrices(A, B *mat.Dense) *mat.Dense {
	rowsA, colsA := A.Dims()
	rowsB, colsB := B.Dims()

	if rowsA != rowsB || colsB != 1 {
		panic("Matrix dimensions do not match for broadcasting")
	}

	// Create a result matrix with the same dimensions as A
	result := mat.NewDense(rowsA, colsA, nil)

	// Subtract B from A, broadcasting B across columns
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			// Divide corresponding element from B (broadcasted across columns)
			result.Set(i, j, A.At(i, j)/B.At(i, 0))
		}
	}

	return result
}

func EncodeStructToJSON(d interface{}, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ") // Pretty-print with 2 spaces of indentation

	// Encode the struct and write to file directly
	err = encoder.Encode(d)
	if err != nil {
		return err
	}

	return nil
}

func WriteJSONBytesToFile(data []byte, filename string) error {
	// Create or truncate the file
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("error creating file: %v", err)
	}
	defer file.Close()

	// Create a buffered writer
	bufferSize := 4 * 1024 * 1024 // 4MB buffer
	writer := bufio.NewWriterSize(file, bufferSize)

	// Write data in chunks
	chunkSize := 1 * 1024 * 1024 // 1MB chunks
	for i := 0; i < len(data); i += chunkSize {
		end := i + chunkSize
		if end > len(data) {
			end = len(data)
		}

		_, err := writer.Write(data[i:end])
		if err != nil {
			return fmt.Errorf("error writing chunk to file: %v", err)
		}

		// Flush the buffer every few chunks to free up memory
		if i%(chunkSize*10) == 0 {
			err = writer.Flush()
			if err != nil {
				return fmt.Errorf("error flushing buffer: %v", err)
			}
		}
	}

	// Final flush to ensure all data is written
	err = writer.Flush()
	if err != nil {
		return fmt.Errorf("error on final flush: %v", err)
	}

	return nil
}

func readImage(i os.DirEntry, imgsPath string, f os.DirEntry) ([]float64, byte) {
	if !i.IsDir() {
		// read imgs and store in slice
		X, err2 := ReadBytes(fmt.Sprintf("%s/%s", imgsPath, i.Name()), false, false)
		if err2 != nil {
			panic(err2)
		}

		y, err3 := strconv.ParseUint(f.Name(), 10, 8)
		if err3 != nil {
			panic(err3)
		}

		return X, byte(y)
	}
	return nil, 0
}

func byteTofloat(b []byte) []float64 {
	var slice []float64
	for i := 0; i < len(b); i++ {
		slice = append(slice, float64(b[i]))
	}
	return slice
}

func ReadBytes(imagePath string, invertColor bool, convertToGrayscale bool) ([]float64, error) {
	imgFile, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}

	defer imgFile.Close()

	imgPng, _ := png.Decode(imgFile)
	if convertToGrayscale {
		imgPng = ConvertIntoGrayscale(imgPng, 28, 28)
	}
	imgBytes := imgPng.(*image.Gray)

	return ScaleValues(imgBytes, invertColor)
}

func ConvertIntoGrayscale(src image.Image, width int, height int) image.Image {
	dst := image.NewGray(image.Rect(0, 0, width, height))
	draw.NearestNeighbor.Scale(dst, dst.Rect, src, src.Bounds(), draw.Over, nil)
	return dst
}

func ScaleValues(data *image.Gray, invertColor bool) ([]float64, error) {
	rows := data.Bounds().Max.X
	cols := data.Bounds().Max.Y

	var result []float64

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// subsequently the scaling_type should be passed as a parameter to the func
			value := data.At(i, j).(color.Gray)
			valueY := value.Y
			if invertColor {
				valueY = 255 - valueY
			}
			scaledValue, err := scaling.Scale(scaling.NEG_ONE_TO_POS_ONE, float64(valueY), 255) // hardcoded maxValue since its an image
			if err != nil {
				return nil, err
			}
			result = append(result, scaledValue)
		}
	}

	return result, nil
}

func NormalizeGrascaleImageData(img image.Image, invertColor bool) ([]float64, error) {
	rows := img.Bounds().Max.X
	cols := img.Bounds().Max.Y

	data := make([]float64, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			gray, ok := img.At(i, j).(color.Gray)
			if ok {
				y := gray.Y
				if invertColor {
					y = 255 - y
				}
				// convert the range into -1.0...1.0
				data[i*rows+j] = (float64(y) - 127.5) / 127.5
			} else {
				return nil, errors.New("cannot take Grayscale color")
			}
		}
	}

	return data, nil
}

func GetDistinctValues(slice []float64) []float64 {
	// Create a map to store unique values
	uniqueMap := make(map[float64]bool)

	// Iterate through the slice and add each value to the map
	for _, value := range slice {
		uniqueMap[value] = true
	}

	// Create a slice to store the unique values
	uniqueSlice := make([]float64, 0, len(uniqueMap))

	// Add all keys from the map to the slice
	for value := range uniqueMap {
		uniqueSlice = append(uniqueSlice, value)
	}

	// Sort the slice for consistent output
	sort.Float64s(uniqueSlice)

	return uniqueSlice
}

func cleanHexString(hexStr string) string {
	// Remove "0x" prefix if present
	hexStr = strings.TrimPrefix(hexStr, "0x")

	// Remove any non-hexadecimal characters
	re := regexp.MustCompile("[^0-9A-Fa-f]")
	hexStr = re.ReplaceAllString(hexStr, "")

	// Ensure the hex string is 16 characters (64 bits)
	for len(hexStr) < 16 {
		hexStr = "0" + hexStr
	}
	return hexStr[:16] // Truncate if longer than 16 characters
}

func HexToFloat64(hexStr string) (float64, error) {
	cleanedHex := cleanHexString(hexStr)

	// Decode hex string to bytes
	bytes, err := hex.DecodeString(cleanedHex)
	if err != nil {
		return 0, fmt.Errorf("failed to decode hex string: %v", err)
	}

	// Convert bytes to uint64
	bits := uint64(0)
	for i, b := range bytes {
		bits |= uint64(b) << (56 - 8*i)
	}

	// Convert uint64 to float64
	return math.Float64frombits(bits), nil
}

func ShuffleSlice[T any](slice []T) []T {
	rnd := rand.New(rand.NewSource(time.Now().UnixNano())) // Seed the random number generator
	rnd.Shuffle(len(slice), func(i, j int) {
		slice[i], slice[j] = slice[j], slice[i]
	})

	return slice
}

func GetRange(n int) []int {
	result := make([]int, n)
	for i := 0; i < n; i++ {
		result[i] = i
	}
	return result
}

func GetBatch[T datamodels.TrainingData | datamodels.ValidationData](data T, step, batchSize int) (*mat.Dense, *mat.Dense) {
	switch type_ := any(data).(type) {
	case datamodels.TrainingData:
		return getBatch(type_.X, type_.Y, step, batchSize)
	case datamodels.ValidationData:
		return getBatch(type_.X, type_.Y, step, batchSize)
	default:
		panic("invalid type")
	}
}

func GetSingleBatch(X *mat.Dense, step, batchSize int) *mat.Dense {
	rows, cols := X.Dims()

	start := step * batchSize
	end := (step + 1) * batchSize

	if end > rows {
		end = rows
	}
	batch_X := mat.DenseCopyOf(X.Slice(start, end, 0, cols))

	return batch_X
}

func getBatch(X *mat.Dense, y *mat.Dense, step, batchSize int) (*mat.Dense, *mat.Dense) {
	rows, cols := X.Dims()
	_, yCols := y.Dims()

	// Define start and end of the slice (batch)
	start := step * batchSize
	end := (step + 1) * batchSize

	// Ensure the end index doesn't exceed the number of rows in X or columns in y
	if end > rows {
		end = rows
	}

	// Slice the X matrix and y vector
	batch_X := mat.DenseCopyOf(X.Slice(start, end, 0, cols))

	if end > yCols {
		end = yCols
	}
	batch_y := mat.DenseCopyOf(y.Slice(0, 1, start, end))

	return batch_X, batch_y
}

func CreateDenseMatrix(rows, cols int, data []float64) *mat.Dense {
	return mat.NewDense(rows, cols, data)
}

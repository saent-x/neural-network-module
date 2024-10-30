package scaling

import (
	"errors"
	"gonum.org/v1/gonum/stat"
	"sort"
)

const (
	NEG_ONE_TO_POS_ONE = 0
	ZERO_TO_ONE        = 1
)

func Scale(scale_type int, value float64, maxValue float64) (float64, error) {
	if scale_type == NEG_ONE_TO_POS_ONE {
		return scale1(value, maxValue), nil
	} else if scale_type == ZERO_TO_ONE {
		return scale2(value, maxValue), nil
	} else {
		return 0, errors.New("scale type error")
	}
}

// scale1 func scales a value from -1 to 1
func scale1(value float64, maxValue float64) float64 {
	halfOfMax := maxValue / 2

	return (value - halfOfMax) / halfOfMax
}

// scale2 func scales a value from 0 to 1
func scale2(value float64, maxValue float64) float64 {
	return value / maxValue
}

func RobustScale(data []float64) ([]float64, error) {
	// Make a sorted copy of the data for quantile calculations
	sortedData := append([]float64(nil), data...) // copy of data
	sort.Float64s(sortedData)                     // sort the copied data

	// Calculate the median of the data
	median := stat.Quantile(0.5, stat.Empirical, sortedData, nil)

	// Calculate the 1st and 3rd quartiles for IQR
	q1 := stat.Quantile(0.25, stat.Empirical, sortedData, nil)
	q3 := stat.Quantile(0.75, stat.Empirical, sortedData, nil)
	iqr := q3 - q1

	// Avoid division by zero if IQR is 0
	if iqr == 0 {
		return nil, errors.New("IQR is zero; data might be constant")
	}

	// Apply robust scaling
	scaledData := make([]float64, len(data))
	for i, v := range data {
		scaledData[i] = (v - median) / iqr
	}

	return scaledData, nil
}
